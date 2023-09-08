from tabpfn import TabPFNClassifier
from models.basemodel import BaseModel

class TabPFNModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            raise NotImplementedError("Does not support")
        elif args.objective == "classification":
            self.model = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)
        elif args.objective == "binary":
            self.model = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)

    def fit(self, X, y, X_val=None, y_val=None):
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

