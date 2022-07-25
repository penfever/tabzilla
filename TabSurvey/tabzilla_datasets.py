# this script is based on the TabSurvey script and function load_data, which returns a specific dataset (X, y).
# instead, this script "prepares" each dataset implemented in our codebase, by doing the following:
# - reading the dataset from a file or online source
# - applying any necessary cleaning (no pre-processing, like encoding or scaling variables)
# - writes each dataset to its own local directory. each dataset directory will contain:
# -- a compressed version of the dataset (X.npy and y.npy)
# -- a json containing metadata
#
# example use - initialize the CaliforniaHousing dataset, write it to a local folder, and then read it into a new TabularDataset object:
#
# >>> from tabzilla_prepare_data import CaliforniaHousing, TabularDataset
# >>> from pathlib import Path
# >>> c = CaliforniaHousing()
# >>> c.name
# 'CaliforniaHousing'
# >>> p = Path("temp/cal-housing").resolve()
# >>> c.write(p)
# >>> c2 = TabularDataset.read(p)
# >>> c2.name
# 'CaliforniaHousing'


import gzip
import json
from pathlib import Path

import numpy as np
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder


class TabularDataset(object):
    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        cat_idx: list,
        target_type: str,
        num_classes: int,
        num_features: int,
        num_instances: int,
        target_encode=False,
    ) -> None:
        """
        name: name of the dataset
        X: matrix of shape (num_instances x num_features)
        y: array of length (num_instances)
        cat_idx: indices of categorical features
        target_type: {"regression", "classification", "binary"}
        num_classes: 1 for regression 2 for binary, and >2 for classification
        num_features: number of features
        num_instances: number of instances
        """
        assert isinstance(X, np.ndarray), "X must be an instance of np.ndarray"
        assert isinstance(y, np.ndarray), "y must be an instance of np.ndarray"
        assert (
            X.shape[0] == num_instances
        ), f"first dimension of X must be equal to num_instances. X has shape {X.shape}"
        assert (
            X.shape[1] == num_features
        ), f"second dimension of X must be equal to num_features. X has shape {X.shape}"
        assert y.shape == (
            num_instances,
        ), f"shape of y must be (num_instances, ). y has shape {y.shape} and num_instances={num_instances}"

        if len(cat_idx) > 0:
            assert (
                max(cat_idx) <= num_features - 1
            ), f"max index in cat_idx is {max(cat_idx)}, but num_features is {num_features}"
        assert target_type in ["regression", "classification", "binary"]

        if target_type == "regression":
            assert num_classes == 1
        elif target_type == "binary":
            assert num_classes == 1
        elif target_type == "classification":
            assert num_classes > 2

        self.name = name
        self.X = X
        self.y = y
        self.cat_idx = cat_idx
        self.target_type = target_type
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_instances = num_instances
        self.target_encode = target_encode

        if target_encode:
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)

        # TODO: move this setting of cat_dims to the preprocessing script, and take cat_dims as a constructor arg
        cat_dims = []
        num_idx = []
        for i in range(self.num_features):
            if self.cat_idx and i in self.cat_idx:
                le = LabelEncoder()
                _ = le.fit_transform(X[:, i])
                # Setting this?
                cat_dims.append(len(le.classes_))
            else:
                num_idx.append(i)

        self.cat_dims = cat_dims

            # TODO: Verify this is not needed
            # # Setting this if classification task
            # if target_type == "classification":
            #     self.num_classes = len(le.classes_)
            #     print("Having", self.num_classes, "classes as target.")

        pass

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "cat_idx": self.cat_idx,
            "target_type": self.target_type,
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "num_instances": self.num_instances,
            "target_encode": self.target_encode,
        }

    @classmethod
    def read(cls, p: Path):
        """read a dataset from a folder"""

        # make sure that all required files exist in the directory
        X_path = p.joinpath("X.npy.gz")
        y_path = p.joinpath("y.npy.gz")
        metadata_path = p.joinpath("metadata.json")

        assert X_path.exists(), f"path to X does not exist: {X_path}"
        assert y_path.exists(), f"path to X does not exist: {y_path}"
        assert metadata_path.exists(), f"path to X does not exist: {metadata_path}"

        # read data
        with gzip.GzipFile(X_path, "r") as f:
            X = np.load(f)
        with gzip.GzipFile(y_path, "r") as f:
            y = np.load(f)

        # read metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return cls(
            metadata["name"],
            X,
            y,
            metadata["cat_idx"],
            metadata["target_type"],
            metadata["num_classes"],
            metadata["num_features"],
            metadata["num_instances"],
        )

    def write(self, p: Path, overwrite=False) -> None:
        """write the dataset to a new folder. this folder cannot already exist"""

        if not overwrite:
            assert ~p.exists(), f"the path {p} already exists."

        # create the folder
        p.mkdir(parents=True, exist_ok=overwrite)

        # write data
        with gzip.GzipFile(p.joinpath("X.npy.gz"), "w") as f:
            np.save(f, self.X)
        with gzip.GzipFile(p.joinpath("y.npy.gz"), "w") as f:
            np.save(f, self.y)

        # write metadata
        with open(p.joinpath("metadata.json"), "w") as f:
            json.dump(self.get_metadata(), f, indent=4)


class CaliforniaHousing(TabularDataset):
    """from sklearn"""

    def __init__(self):
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        super().__init__(
            "CaliforniaHousing",
            X,
            y,
            [],
            "regression",
            1,
            8,
            len(y),
        )