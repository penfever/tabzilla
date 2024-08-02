from tabpfn import TabPFNClassifier
from models.basemodel import BaseModel

# :param model_string: Name of the model. Used first to check if the model is already in memory, and if not, 
#         tries to load a model with that name from the models_diff directory. It looks for files named as 
#         follows: "prior_diff_real_checkpoint" + model_string + "_n_0_epoch_e.cpkt", where e can be a number 
#         between 100 and 0, and is checked in a descending order. 
#_multiclass_11_13_2023_16_42_28

class TabPFNModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            raise NotImplementedError("Does not support")
        elif args.objective == "classification":
            self.model = TabPFNClassifier(device='cuda', N_ensemble_configurations=32, model_string=args.checkpoint_path)
        elif args.objective == "binary":
            self.model = TabPFNClassifier(device='cuda', N_ensemble_configurations=32, model_string=args.checkpoint_path)

    def fit(self, X, y, X_val=None, y_val=None):
        # print("Train data shape: ", X.shape)
        # print("Train data:", X[:10, ...])
        self.model.fit(X, y, overwrite_warning=True)
        return [], []

    def predict_helper(self, X):
        return self.model.predict(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = dict()
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        params = dict()
        return params

    @classmethod
    def default_parameters(cls):
        params = dict()
        return params

    def get_classes(self):
        return self.model.classes_

