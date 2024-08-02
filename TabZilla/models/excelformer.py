# The ExcelFormer model.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from torch import einsum
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.basemodel_torch import BaseModelTorch
from models.excelformer_lib import ExcelFormer as ExcelFormerArch
import models.excelformer_lib.lib as excelformer_lib 

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, log_loss
import sklearn

from category_encoders import CatBoostEncoder
import warnings

"""
    ExcelFormer: A neural network surpassing GBDTs on tabular data
    (https://arxiv.org/abs/2301.02819)
"""

class ExcelFormer(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.cat_idx:
            num_idx = list(set(range(args.num_features)) - set(args.cat_idx))
            cat_dims = np.array(args.cat_dims).astype(int)
        else:
            cat_dims = []
            num_idx = list(range(args.num_features))
        
        self.num_idx = num_idx
        self.cat_dims = cat_dims

        # Decreasing some hyperparameter to cope with memory issues
        # dim = self.params["dim"] if args.num_features < 50 else 8
        # self.batch_size = self.args.batch_size if args.num_features < 50 else 64

        print("Using dim %d and batch size %d" % (self.params["dim"], args.batch_size))
        #args.num_features
        model_args = {
            # 'd_numerical': len(num_idx),
            'd_numerical': args.num_features,
            'd_out': args.num_classes,
            #'categories': cat_dims,
            'categories': None, # https://github.com/WhatAShot/ExcelFormer/blob/17f70526390e70390bb8c8ec3850697eb730f9cd/run_default_config_excel.py#L192
            "prenormalization": True, # true or false, perform BETTER on a few datasets with no prenormalization 
            'kv_compression': None,
            'kv_compression_sharing': None,
            'token_bias': True,
            'ffn_dropout': 0., 
            'attention_dropout':  0.3,
            'residual_dropout': self.params["dropout"],
            'n_layers': self.params["depth"], # 3 
            'n_heads': self.params["heads"], # 32
            'd_token': self.params["dim"], # 256
            'init_scale': 0.01, # param for the Attenuated Initialization
        }

        self.model = ExcelFormerArch(**model_args)

        if args.data_parallel:
            self.model = nn.DataParallel(self.model)

        self.normalizer = sklearn.preprocessing.QuantileTransformer(
                output_distribution='normal',
                n_quantiles=10,
                subsample=1e9,
                random_state=self.args.hparam_seed,
        )

    def make_excelformer_dataset(self, X, y, is_train=False):
        warnings.filterwarnings("ignore")
        X_num = X[:, self.num_idx]
        
        if self.args.cat_idx:
            X_cat = X[:, self.args.cat_idx]
        else:
            X_cat = None

        X_num = X_num.astype(np.float32, casting='unsafe')
        # X_cat = X_cat.astype(np.float32, casting='unsafe') if X_cat is not None else None


        if X_num.shape[1] > 0:
            # fill numeric NaNs using nanmean
            X_num = np.nan_to_num(X_num, nan=np.nanmean(X_num, axis=0))

            # quantile normalization is standard in ExcelFormer
            if is_train:
                X_num = self.normalizer.fit_transform(X_num)
            else:
                X_num = self.normalizer.transform(X_num)
        

        # ExcelFormer adds noise to the data
        if is_train:
            noise = 1e-3
            stds = np.std(X_num, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_num = X_num + noise_std * np.random.default_rng(self.args.hparam_seed).standard_normal(
                X_num.shape
            )

        # fill categorical NaNs with the most frequent value
        if X_cat is not None:
            for i in range(X_cat.shape[1]):
                col = X_cat[:, i]
                try:
                    col[np.isnan(col)] = np.nanmax(col)
                except:
                    pass
        
        if X_cat is not None:
            if is_train:
                self.enc = CatBoostEncoder(
                    cols=list(range(len(self.cat_dims))), 
                    return_df=False
                ).fit(X_cat, y)
            # directly regard catgorical features as numerical
            X_num = np.concatenate([self.enc.transform(X_cat), X_num], axis=1)
            X_cat = None
        
        # #np to tensor
        X_num = torch.tensor(X_num, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long if self.args.objective == "classification" else torch.float32)

        ds = TensorDataset(X_num, y)
        warnings.filterwarnings("default")
        return ds
    
    def get_loss(self, loss_fn, X_num, y):
        if self.params['mix_type'] == 'none':
            preds = self.model(X_num, None, mixup=False)
            loss = loss_fn(preds, y)
        else:
            preds, feat_masks, shuffled_ids = self.model(X_num, None, mixup=True, mtype=self.params['mix_type'])
            # print(preds, feat_masks, shuffled_ids)
            # exit()
            if self.params['mix_type'] == 'hidden_mix':
                lambdas = feat_masks.float()
                lambdas2 = 1 - lambdas
            elif self.params['mix_type'] == 'niave_mix':
                lambdas = feat_masks.float()
                lambdas2 = 1 - lambdas
            if self.args.objective == "regression":
                mix_y = lambdas * y + lambdas2 * y[shuffled_ids]
                loss = loss_fn(preds, mix_y)
            else:
                loss = lambdas * loss_fn(preds, y) + lambdas2 * loss_fn(preds, y[shuffled_ids])
                loss = loss.mean()
        return loss

    def fit(self, X, y, X_val=None, y_val=None):

        # optimizer
        # def needs_wd(name):
        #     return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

        # parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
        # parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            # [
            #     {'params': parameters_with_wd, 'weight_decay': 0.},
            #     {'params': parameters_without_wd, 'weight_decay': 0},
            # ],
            lr=self.params["lr"],
        )

        self.model.to(self.device)

        if self.args.objective == "binary":
            if self.params['mix_type'] == 'none':
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif self.args.objective == "classification":
            if self.params['mix_type'] == 'none':
                loss_fn = nn.CrossEntropyLoss()
            else:
                loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            loss_fn = nn.MSELoss()

        #sanity check with a simple logistic regression
        # from sklearn.linear_model import LogisticRegression

        # if self.args.objective != "regression":
        #     clf = LogisticRegression(random_state=0).fit(X, y)
        #     print("Logistic regression score:", round(clf.score(X_val, y_val), 4))


        train_ds = self.make_excelformer_dataset(X, y, is_train=True)

        train_loader = DataLoader(
            train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=2
        )

        #one-hot encoding of y_val for classification
        if self.args.objective == "classification":
            y_val = np.eye(self.args.num_classes)[y_val]

        # we use AUC for binary classification, Accuracy for multi-class classification, RMSE for regression
        # metric = 'roc_auc' if self.args.objective == "binary" else 'score'
        metric = roc_auc_score if self.args.objective == "binary" else accuracy_score

        n_epochs = 500 # default max training epoch

        # warmup and lr scheduler
        warm_up = 10 # warm up epoch
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs - warm_up) # lr decay
        max_lr = 1e-4

        # report_frequency = len(ys['train']) // batch_size // 3

        loss_history = []
        val_loss_history = []

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        # early stop
        no_improvement = 0
        EARLY_STOP = self.args.early_stopping_rounds
        report_frequency = len(train_loader) // 2

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            # warm up lr
            if warm_up > 0 and epoch <= warm_up:
                lr = max_lr * epoch / warm_up
                # print(f'warm up ({epoch}/{warm_up})')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                scheduler.step()
            for iteration, batch in enumerate(train_loader):
                #either x_num, x_cat, y or x_num, y
                x_num, x_cat, y = (
                    (batch[0], None, batch[1])
                    if len(batch) == 2
                    else batch
                )
                if x_cat is None:
                    x_num, y = x_num.to(self.device), y.to(self.device)
                else:
                    x_num, x_cat, y = x_num.to(self.device), x_cat.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                
                loss = self.get_loss(loss_fn, x_num, y)
                loss_history.append(loss.item())
                loss.backward()
                optimizer.step()
            val_preds = self.predict_helper(X_val)
            val_loss = log_loss(y_val, val_preds) if self.args.objective != "regression" else mean_squared_error(y_val, val_preds)
            val_loss_history.append(val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch
                no_improvement = 0
                print(f'Epoch {epoch:03d} | Min Val Loss: {min_val_loss:.4f} \n', end='')
            else:
                no_improvement += 1
            if no_improvement >= EARLY_STOP:
                print(f'No improvement for {EARLY_STOP} epochs, early stopping')
                break

        return loss_history, val_loss_history

    @torch.inference_mode()
    def predict_helper(self, X):
        self.model.eval()
        val_ds = self.make_excelformer_dataset(X, np.zeros(X.shape[0]))

        val_loader = DataLoader(
            val_ds, batch_size=self.args.val_batch_size, shuffle=False, num_workers=2
        )

        preds = []

        for iteration, batch in enumerate(val_loader):
            #either x_num, x_cat, y or x_num, y
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            if x_cat is None:
                x_num, y = x_num.to(self.device), y.to(self.device)
            else:
                x_num, x_cat, y = x_num.to(self.device), x_cat.to(self.device), y.to(self.device)
            output = self.model(x_num, None)

            if self.args.objective == "classification":
                preds.append(F.softmax(output, dim=1))
            elif self.args.objective == "binary":
                preds.append(torch.sigmoid(output))
            else:
                preds.append(output)
        ret_preds = torch.cat(preds).cpu().detach().numpy()
        if ret_preds.ndim == 1:
            ret_preds = np.expand_dims(ret_preds, axis=1)
        return ret_preds

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "dim": trial.suggest_categorical("dim", [32, 64, 128, 256]),
            "depth": trial.suggest_categorical("depth", [2, 3, 4, 5]),
            "heads": trial.suggest_categorical("heads", [4, 8, 16, 32]),
            "dropout": trial.suggest_categorical(
                "dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            ),
            "mix_type" : trial.suggest_categorical("mix_type", ["hidden_mix", "niave_mix", "none"]),
            "lr" : trial.suggest_categorical("lr", [1e-3, 5e-4, 1e-4, 3e-5]),
        }
        return params

    # TabZilla: add function for seeded random params and default params
    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "dim": rs.choice([32, 64, 128, 256]),
            "depth": rs.choice([2, 3, 4, 5]),
            "heads": rs.choice([4, 8, 16, 32]),
            "dropout": rs.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            "mix_type" : rs.choice(["hidden_mix", "niave_mix", "none"]),
            "lr" : rs.choice([1e-3, 5e-4, 1e-4, 3e-5]),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {
            "dim": 256,
            "depth": 3,
            "heads": 32,
            "dropout": 0.3,
            # "mix_type" : "none",
            "mix_type" : "hidden_mix",
            # "lr" : 1e-2,
            "lr" : 1e-4,
        }
        return params
