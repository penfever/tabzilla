import gzip
import json
import os
import shutil
import signal
import time
import gc
from contextlib import contextmanager
from collections import namedtuple
from typing import NamedTuple
from pathlib import Path

import numpy as np
from models.basemodel import BaseModel
from tabzilla_data_processing import process_data
from tabzilla_datasets import TabularDataset
from utils.scorer import BinScorer, ClassScorer, RegScorer
from utils.timer import Timer
import scipy.stats as stats

def free_memory(sleep_time=0.1):
    """ Black magic function to free torch memory and some jupyter whims """
    import torch
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_filepath(name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD.<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    return (name + "_%s." + extension) % timestr


def is_jsonable(x, cls=None):
    try:
        json.dumps(x, cls=cls)
        return True
    except (TypeError, OverflowError):
        return False


def get_scorer(objective):
    if objective == "regression":
        return RegScorer()
    elif objective == "classification":
        return ClassScorer()
    elif objective == "binary":
        return BinScorer()
    else:
        raise NotImplementedError('No scorer for "' + objective + '" implemented')


class ExperimentResult:
    """
    container class for an experiment result.

    attributes:
    - dataset(TabularDataset): a dataset object
    - scaler(str): scaler for numerical features
    - model(BaseModel): the model trained & evaluated on the dataset
    - timers(dict[Timer]): timers for training and evaluating model
    - scorers(dict): scorer objects for train, test, and val sets
    - predictions(dict): output of the model on the dataset. keys = "train", "test", "val"
    - probabilities(dict): probabilities of predicted class (only for classification problems)
    - ground_truth(dict): ground truth for each prediction, stored here just for convenience.
    - hparam_source(str): a string describing how the hyperparameters were generated
    - trial_number(int): trial number

    attributes "predictions", "probabilities", and "ground_truth" each have the same shape as the lists in dataset.split_indeces.
    """

    def __init__(
        self,
        dataset,
        scaler,
        model,
        timers,
        scorers,
        predictions,
        probabilities,
        ground_truth,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.model = model
        self.timers = timers
        self.scorers = scorers
        self.predictions = predictions
        self.probabilities = probabilities
        self.ground_truth = ground_truth

        # we will set these after initialization
        self.hparam_source = None
        self.trial_number = None
        self.experiemnt_args = None
        self.exception = None

    def write(self, filepath_base, compress=False):
        """
        write two files:
        - one with the results from the trial, including metadata and performance, and
        - one with all metadata, all predictions, ground truth, and split indices.
        """

        # create a dict with all output we want to store
        result_dict = {
            "dataset": self.dataset.get_metadata(),
            "scaler": self.scaler,
            "model": self.model.get_metadata(),
            "experiemnt_args": self.experiment_args,
            "hparam_source": self.hparam_source,
            "trial_number": self.trial_number,
            "exception": str(self.exception),
            "timers": {name: timer.save_times for name, timer in self.timers.items()},
            "scorers": {
                name: scorer.get_results() for name, scorer in self.scorers.items()
            },
        }

        # add the predictions (lots of data) to a new dict
        prediction_dict = result_dict.copy()

        prediction_dict["predictions"] = self.predictions
        prediction_dict["probabilities"] = self.probabilities
        prediction_dict["ground_truth"] = self.ground_truth
        prediction_dict["splits"] = [
            {key: list(val.tolist()) for key, val in split.items()}
            for split in self.dataset.split_indeces
        ]

        # write results
        for k, v in result_dict.items():
            if not is_jsonable(v, cls=NpEncoder):
                raise Exception(
                    f"writing results: value at key '{k}' is not json serializable: {v}"
                )

        write_dict_to_json(
            result_dict,
            Path(str(filepath_base) + "_results.json"),
            compress=compress,
            cls=NpEncoder,
        )

        # write predictions
        for k, v in prediction_dict.items():
            if not is_jsonable(v, cls=NpEncoder):
                raise Exception(
                    f"writing predictions: value at key '{k}' is not json serializable: {v}"
                )

        write_dict_to_json(
            prediction_dict,
            Path(str(filepath_base) + "_predictions.json"),
            compress=compress,
            cls=NpEncoder,
        )


class TimeoutException(Exception):
    pass

def cross_validation(model: BaseModel, dataset: TabularDataset, time_limit: int, scaler: str, args : NamedTuple) -> ExperimentResult:
    """
    takes a BaseModel and TabularDataset as input, and trains and evaluates the model using cross validation with all
    folds specified in the dataset property split_indeces. Time limit is checked after each fold, and an exception is raised.
    Scaler is passed to tabzilla_data_processing.process_data()

    returns an ExperimentResult object, which contains all metadata and results from the cross validation run, including:
    - evlaution objects for the validation and test sets
    - predictions and prediction probabilities for all data points in each fold.
    - runtimes for training and evaluation, for each fold
    """

    # Record some statistics and metrics
    # create a scorer & timer object for the train, val, and test sets
    scorers = {
        "train": get_scorer(dataset.target_type),
        "val": get_scorer(dataset.target_type),
        "test": get_scorer(dataset.target_type),
    }
    timers = {
        "train": Timer(),
        "val": Timer(),
        "test": Timer(),
        "train-eval": Timer(),
    }

    # store predictions and class probabilities. probs will be None for regression problems.
    # these have the same dimension as train_index, test_index, and val_index
    predictions = {
        "train": [],
        "val": [],
        "test": [],
    }
    probabilities = {
        "train": [],
        "val": [],
        "test": [],
    }
    ground_truth = {
        "train": [],
        "val": [],
        "test": [],
    }

    start_time = time.time()
    print("Current model: ", model)
    # iterate over all train/val/test splits in the dataset property split_indeces
    for i, split_dictionary in enumerate(dataset.split_indeces):
        if time.time() - start_time > time_limit:
            raise TimeoutException(f"time limit of {time_limit}s reached during fold {i}")

        train_index = split_dictionary["train"]
        val_index = split_dictionary["val"]
        test_index = split_dictionary["test"]

        # run pre-processing & split data (list of numpy arrays of length num_ensembles)
        processed_data = process_data(
            dataset,
            train_index,
            val_index,
            test_index,
            verbose=False,
            scaler=scaler,
            one_hot_encode=False,
            args=args,
        )
        X_train, y_train = processed_data["data_train"]
        X_val, y_val = processed_data["data_val"]
        X_test, y_test = processed_data["data_test"]
        
        train_predictions_list = []
        train_probs_list = []
        val_predictions_list = []
        val_probs_list = []
        test_predictions_list = []
        test_probs_list = []

        for j in range(args.num_ensembles):
            try:
                free_memory()
            except Exception as e:
                print(f"Failed to free memory, error message was: \n {e}")
            print("Ensemble iteration ", j+1, " of ", args.num_ensembles)
            # Create a new unfitted version of the model
            curr_model = model.clone()

            # Train model
            timers["train"].start()
            print("Sizes of datasets: ", len(X_train[j]), len(X_val[j]), len(X_test[j]))
            # loss history can be saved if needed
            print("Fitting model, iteration ", i, " of ", len(dataset.split_indeces))
            loss_history, val_loss_history = curr_model.fit(
                X_train[j],
                y_train[j],
                X_val[j],
                y_val[j],
            )
            print("Done fitting model, history is: ", loss_history, val_loss_history)
            timers["train"].end()
            n_test_rows = args.subset_rows
            # Experimenting with limiting TabPFN test rows to save VRAM
            # if "TabPFN" in str(args.model_name):
            #     print("Limiting TabPFN test rows to 1000")
            #     n_test_rows = 1000
            # evaluate on train set
            timers["train-eval"].start()
            train_predictions, train_probs = curr_model.predict_wrapper(X_train[j], n_test_rows)
            train_predictions_list.append(train_predictions)
            train_probs_list.append(train_probs)
            timers["train-eval"].end()
            print("Train eval done")
            # evaluate on val set
            timers["val"].start()
            val_predictions, val_probs = curr_model.predict_wrapper(X_val[j], n_test_rows)
            val_predictions_list.append(val_predictions)
            val_probs_list.append(val_probs)
            timers["val"].end()
            print("Val eval done")
            # evaluate on test set
            timers["test"].start()
            test_predictions, test_probs = curr_model.predict_wrapper(X_test[j], n_test_rows)
            test_predictions_list.append(test_predictions)
            test_probs_list.append(test_probs)
            timers["test"].end()
            print("Test eval done")
            extra_scorer_args = {}
            if dataset.target_type == "classification":
                extra_scorer_args["labels"] = range(dataset.num_classes)
        # Get majority vote
        train_predictions = stats.mode(np.array(train_predictions_list), axis=0).mode.reshape(-1)
        train_probs = np.mean(np.array(train_probs_list), axis=0)
        val_predictions = stats.mode(np.array(val_predictions_list), axis=0).mode.reshape(-1)
        val_probs = np.mean(np.array(val_probs_list), axis=0)
        test_predictions = stats.mode(np.array(test_predictions_list), axis=0).mode.reshape(-1)
        test_probs = np.mean(np.array(test_probs_list), axis=0)
        # evaluate on train, val, and test sets
        scorers["train"].eval(
            y_train[j], train_predictions, train_probs, **extra_scorer_args
        )
        scorers["val"].eval(y_val[j], val_predictions, val_probs, **extra_scorer_args)
        scorers["test"].eval(y_test[j], test_predictions, test_probs, **extra_scorer_args)

        # store predictions & ground truth

        # train
        predictions["train"].append(train_predictions.tolist())
        probabilities["train"].append(train_probs.tolist())
        ground_truth["train"].append(y_train[j].tolist())

        # val
        predictions["val"].append(val_predictions.tolist())
        probabilities["val"].append(val_probs.tolist())
        ground_truth["val"].append(y_val[j].tolist())

        # test
        predictions["test"].append(test_predictions.tolist())
        probabilities["test"].append(test_probs.tolist())
        ground_truth["test"].append(y_test[j].tolist())
        print("Sample accuracy scores from test set splits: ", scorers["test"].accs)

    return ExperimentResult(
        dataset=dataset,
        scaler=scaler,
        model=model,
        timers=timers,
        scorers=scorers,
        predictions=predictions,
        probabilities=probabilities,
        ground_truth=ground_truth,
    )


def write_dict_to_json(x: dict, filepath: Path, compress=False, cls=None):
    assert not filepath.is_file(), f"file already exists: {filepath}"
    assert filepath.parent.is_dir(), f"directory does not exist: {filepath.parent}"
    if not compress:
        with filepath.open("w", encoding="UTF-8") as f:
            json.dump(x, f, cls=cls)
    else:
        with gzip.open(str(filepath) + ".gz", "wb") as f:
            f.write(json.dumps(x, cls=cls).encode("UTF-8"))


def make_archive(source, destination):
    """
    a helper function because shutil.make_archive is too confusing on its own. adapted from:
    http://www.seanbehan.com/how-to-use-python-shutil-make_archive-to-zip-up-a-directory-recursively-including-the-root-folder/
    zip the folder at "source" and write it to the file at "destination". the file type is read from arg "destination"

    example use:
    > make_archive("/source/directory", "/new/directory/archive.zip")
    """

    base = os.path.basename(destination)
    name = base.split(".")[0]
    format = base.split(".")[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move("%s.%s" % (name, format), destination)


import configargparse
import yaml

# the parsers below are based on the TabSurvey parsers in utils.py


def get_experiment_parser():
    """parser for experiment arguments"""

    experiment_parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    experiment_parser.add(
        "-experiment_config",
        required=True,
        is_config_file=True,
        help="config file for arg parser",
    )
    experiment_parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="directory where experiment results will be written.",
    )
    experiment_parser.add(
        "--use_gpu", action="store_true", help="Set to true if GPU is available"
    )
    experiment_parser.add(
        "--gpu_ids",
        type=int,
        action="append",
        help="IDs of the GPUs used when data_parallel is true",
    )
    experiment_parser.add(
        "--data_parallel",
        action="store_true",
        help="Distribute the training over multiple GPUs",
    )
    experiment_parser.add(
        "--n_random_trials",
        type=int,
        default=10,
        help="Number of trials of random hyperparameter search to run",
    )
    experiment_parser.add(
        "--hparam_seed",
        type=int,
        default=0,
        help="Random seed for generating random hyperparameters. passed to optuna RandomSampler.",
    )
    experiment_parser.add(
        "--n_opt_trials",
        type=int,
        default=10,
        help="Number of trials of hyperparameter optimization to run",
    )
    experiment_parser.add(
        "--batch_size", type=int, default=128, help="Batch size used for training"
    )
    experiment_parser.add(
        "--val_batch_size",
        type=int,
        default=128,
        help="Batch size used for training and testing",
    )
    experiment_parser.add(
        "--scale_numerical_features",
        type=str,
        choices=["None", "Quantile", "Standard"],
        default="None",
        help="Specify scaler for numerical features. Applied during data processing, prior to training and inference.",
    )
    experiment_parser.add(
        "--early_stopping_rounds",
        type=int,
        default=20,
        help="Number of rounds before early stopping applies.",
    )
    experiment_parser.add(
        "--epochs", type=int, default=1000, help="Max number of epochs to train."
    )
    experiment_parser.add(
        "--logging_period",
        type=int,
        default=100,
        help="Number of iteration after which validation is printed.",
    )
    experiment_parser.add(
        "--experiment_time_limit",
        type=int,
        default=10,
        help="Time limit for experiment, in seconds.",
    )
    experiment_parser.add(
        "--trial_time_limit",
        type=int,
        default=10,
        help="Time limit for each train/test trial, in seconds.",
    )
    experiment_parser.add(
        "--subset_rows",
        type=int,
        default=-1,
        help="Number of rows to use for training and testing. -1 means use all rows.",
    )
    experiment_parser.add(
        "--subset_features",
        type=int,
        default=-1,
        help="Number of features to use for training and testing. -1 means use all features.",
    )
    experiment_parser.add(
        "--subset_rows_method",
        type=str,
        choices=["random", "first", "kmeans", "coreset", "closest"],
        default="random",
        help="Method for selecting rows, one of 'random', 'first', 'kmeans', 'coreset', 'closest'.",
    )
    experiment_parser.add(
        "--subset_features_method",
        type=str,
        choices=["random", "first", "mutual_information"],
        default="random",
        help="Method for selecting features. 'random' means select randomly, 'first' means select the first features, 'mutual information' wraps sklearn's mutual_info_classif.",
    )
    experiment_parser.add(
        "--y_equalizer",
        type=str,
        choices=["none", "equal", "proportion"],
        default="none",
        help="Method for equalizing the number of samples in each class. 'none' means do nothing, 'equal' means equalize the number of samples in each class, 'proportion' means equalize the proportion of samples in each class.",
    )
    experiment_parser.add(
        "--num_ensembles",
        type=int,
        default=1,
        help="How many times to train the model. The model will be trained from scratch each time.",
    )
    return experiment_parser
