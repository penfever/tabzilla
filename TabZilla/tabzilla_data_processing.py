import numpy as np
import time
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

class CoresetSampler:
    def __init__(self, number_of_set_points, number_of_starting_points, rand_seed):
        self.number_of_set_points = number_of_set_points
        self.number_of_starting_points = number_of_starting_points
        self.rng = np.random.default_rng(rand_seed)

    def _compute_batchwise_differences(self, a, b):
        return np.sum((a[:, None] - b) ** 2, axis=-1)

    def _compute_greedy_coreset_indices(self, features):
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = self.rng.choice(len(features), number_of_starting_points, replace=False).tolist()
        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = np.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = self.number_of_set_points

        for _ in range(num_coreset_samples):
            select_idx = np.argmax(approximate_coreset_anchor_distances)
            coreset_indices.append(select_idx)
            coreset_select_distance = self._compute_batchwise_differences(
                features, features[select_idx : select_idx + 1]
            )
            approximate_coreset_anchor_distances = np.concatenate(
                [approximate_coreset_anchor_distances, coreset_select_distance],
                axis=-1,
            )
            approximate_coreset_anchor_distances = np.min(
                approximate_coreset_anchor_distances, axis=1
            ).reshape(-1, 1)

        return np.array(coreset_indices)

class SubsetMaker(object):

    def __init__(self, subset_features, subset_rows, subset_features_method, subset_rows_method, y_equalizer):
        self.subset_features = subset_features
        self.subset_rows = subset_rows
        self.subset_features_method = subset_features_method
        self.subset_rows_method = subset_rows_method
        self.y_equalizer = y_equalizer
        self.row_selector = None
        self.feature_selector = None
        self.seeds_seen = set()

    def random_subset(self, X, y, action=[], rand_seed=0):
        print("Random seed: ", rand_seed)
        rng = np.random.default_rng(rand_seed)
        if 'rows' in action:
            row_indices = rng.choice(X.shape[0], self.subset_rows, replace=False)
        else:
            row_indices = np.arange(X.shape[0])
        if 'features' in action:
            feature_indices = rng.choice(X.shape[1], self.subset_features, replace=False)
        else:
            feature_indices = np.arange(X.shape[1])
        return X[row_indices[:, None], feature_indices], y[row_indices]
    
    def first_subset(self, X, y, action=[]):
        if 'rows' in action:
            row_indices = np.arange(self.subset_rows)
        else:
            row_indices = np.arange(X.shape[0])
        if 'features' in action:
            feature_indices = np.arange(self.subset_features)
        else:
            feature_indices = np.arange(X.shape[1])
        return X[row_indices[:, None], feature_indices], y[row_indices]
    
    def mutual_information_subset(self, X, y, action='features', split='train'):
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if split == 'train':
            #NOTE: we are only fitting on the first split we see to save time here
            if getattr(self, 'feature_selector', None) is None:
                print("Fitting mutual information feature selector ...")
                #start the timer
                timer = time.time()
                self.feature_selector = SelectKBest(mutual_info_classif, k=self.subset_features)
                X = self.feature_selector.fit_transform(X, y)
                print(f"Done fitting mutual information feature selector in {round(time.time() - timer, 1)} seconds")
            else:
                X = self.feature_selector.transform(X)
            return X, y
        else:
            X = self.feature_selector.transform(X)
            return X, y

    def K_means_sketch(self, X, k, fit_first_only=False, rand_seed=0):
        print("Random seed: ", rand_seed)
        # This function returns the indices of the k samples that are the closest to the k-means centroids
        import faiss
        X = np.ascontiguousarray(X, dtype=np.float32)
        if fit_first_only and getattr(self, 'kmeans', None) is None:
            print("Fitting kmeans sketch  ...")
            #start the timer
            timer = time.time()
            self.kmeans = faiss.Kmeans(X.shape[1], k, niter=15, verbose=False)
            self.kmeans.train(X)
            print(f"Done fitting in {round(time.time() - timer, 1)} seconds")
        elif fit_first_only and getattr(self, 'kmeans', None) is not None:
            pass
        else:
            self.kmeans = faiss.Kmeans(X.shape[1], k, niter=15, verbose=False)
            self.kmeans.train(X)
        cluster_centers = self.kmeans.centroids
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(cluster_centers)
        s_val = min(k * rand_seed, index.ntotal)
        _, indices = index.search(cluster_centers, s_val)
        assert rand_seed > 0, "Random seed for closest sketch must be greater than 0"
        if rand_seed == 1 or (rand_seed-1)*k > len(indices) - k - 1:
            return indices.reshape(-1)
        else:
            return indices[:, (rand_seed-1)*k:].reshape(-1)

    def coreset_sketch(self, X, k, rand_seed=0):
        # This function returns the indices of the k samples that are a greedy coreset
        number_of_set_points = k  # Number of set points for the greedy coreset
        number_of_starting_points = 5  # Number of starting points for the greedy coreset

        sampler = CoresetSampler(number_of_set_points, number_of_starting_points, rand_seed)
        indices = sampler._compute_greedy_coreset_indices(X)
        return indices


    def closest_sketch(self, X, k, X_val, rand_seed=1):
        # This function returns the indices of the k samples that are the closest to the samples in X_val
        # Using faiss
        import faiss
        X = np.ascontiguousarray(X, dtype=np.float32)
        X_val = np.ascontiguousarray(X_val, dtype=np.float32)
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        target_size = min(k * rand_seed, X_val.shape[0])
        try:
            X_val_reshaped = X_val.reshape(1, -1).astype(np.float32)
            _, indices = index.search(X_val_reshaped, target_size)
        except:
            _, indices = index.search(X_val, target_size)
        assert rand_seed > 0, "Random seed for closest sketch must be greater than 0"
        if rand_seed == 1:
            return indices.reshape(-1)
        else:
            return indices[:, (rand_seed-1)*k:].reshape(-1)

    def sketch(self, X, y, sketch_size, X_val=None, rand_seed=0):
        self.seeds_seen.add(rand_seed)
        if self.subset_rows_method == "random":
            rng = np.random.default_rng(rand_seed)
            indices = rng.choice(X.shape[0], sketch_size, replace=False)
        elif self.subset_rows_method == "first":
            indices = np.arange(sketch_size)
        elif self.subset_rows_method == "kmeans":
            indices = self.K_means_sketch(X, sketch_size, fit_first_only=False, rand_seed=len(self.seeds_seen))
        elif self.subset_rows_method == "coreset":
            indices = self.coreset_sketch(X, sketch_size, rand_seed=rand_seed)
        elif self.subset_rows_method == "closest":
            indices = self.closest_sketch(X, sketch_size, X_val, rand_seed=len(self.seeds_seen))
        else:
            raise NotImplementedError("Sketch type not implemented")

        return indices
    
    def make_subset(
        self,
        X,
        y,
        X_val=None,
        split='train',
        num_classes=None,
        rand_seed=0,
        action=['features', 'rows'],
    ):
        """
        Make a subset of the data matrix X, with subset_features features and subset_rows rows.
        :param X: data matrix
        :param y: labels
        :param subset_features: number of features to keep
        :param subset_rows: number of rows to keep
        :param subset_features_method: method to use for selecting features
        :param subset_rows_method: method to use for selecting rows
        :return: subset of X, y
        """
        if 'features' in action:
            print(f"making {self.subset_features}-sized subset of {X.shape[1]} features ...")
            if self.subset_features_method == "random":
                X, y = self.random_subset(X, y, action=['features'], rand_seed=rand_seed)
            elif self.subset_features_method == "first":
                X, y = self.first_subset(X, y, action=['features'])
            elif self.subset_features_method == "mutual_information":
                X, y = self.mutual_information_subset(X, y, action='features', split=split)
            else:
                raise ValueError(f"subset_features_method not recognized: {self.subset_features_method}")
        if 'rows' in action:
            print(f"making {self.subset_rows}-sized subset of {X.shape[0]} rows ...")
            if self.y_equalizer == "none":
                indices = self.sketch(X, y, self.subset_rows, X_val=X_val, rand_seed=rand_seed)
            else:
                # Fix for how TabZilla handles binary
                if num_classes == 1:
                    num_classes = 2
                y_vals = np.arange(num_classes)
                print("num classes in dataset: ", num_classes)
                indices = []
                for i in range(len(y_vals)):
                    relevant_indices = np.where(y == y_vals[i])[0]
                    if len(relevant_indices) == 0:
                        continue
                    # if i == len(y_vals) - 1:
                    #     samples_per_index = self.subset_rows - len(indices)
                    # else:
                    if self.y_equalizer == "equal":
                        equal_size = int(self.subset_rows / len(y_vals))
                        if equal_size > len(relevant_indices):
                            print("Warning: can't get equal-sized subset of each class. Instead using all available samples from this class")
                            samples_per_index = len(relevant_indices)
                        else:
                            samples_per_index = equal_size
                    elif self.y_equalizer == "proportion":
                        y_val_proportion = len(relevant_indices) / len(y)
                        # print("int(np.ceil(self.subset_rows*y_val_proportion))", int(np.ceil(self.subset_rows*y_val_proportion)))
                        samples_per_index = min(int(np.ceil(self.subset_rows*y_val_proportion)) + 1, len(relevant_indices))
                    else:
                        raise ValueError(f"y_equalizer not recognized: {self.y_equalizer}")

                    print(f"Taking {samples_per_index} samples from class {i}")
                    # print("len(relevant_indices)", len(relevant_indices))
                    indices_per_index = self.sketch(X[relevant_indices], y[relevant_indices], samples_per_index, X_val=X_val, rand_seed=rand_seed)

                    indices.extend(relevant_indices[indices_per_index[:len(relevant_indices)]])
            X, y = X[indices], y[indices]
            # print(f"shape(X) = {X.shape}")
            # print(f"shape(y) = {y.shape}")
            # print(f"number of classes = {len(np.unique(y))}")

        return X, y

def process_data(
    dataset,
    train_index,
    val_index,
    test_index,
    verbose=False,
    scaler="None",
    one_hot_encode=False,
    impute=True,
    args=None,
):
    
    # validate the scaler
    assert scaler in ["None", "Quantile"], f"scaler not recognized: {scaler}"

    if scaler == "Quantile":
        scaler_function = QuantileTransformer(n_quantiles=min(len(train_index), 1000))  # use either 1000 quantiles or num. training instances, whichever is smaller


    num_mask = np.ones(dataset.X.shape[1], dtype=int)
    num_mask[dataset.cat_idx] = 0
    # TODO: Remove this assertion after sufficient testing
    assert num_mask.sum() + len(dataset.cat_idx) == dataset.X.shape[1]

    X_train, y_train = dataset.X[train_index], dataset.y[train_index]
    X_val, y_val = dataset.X[val_index], dataset.y[val_index]
    X_test, y_test = dataset.X[test_index], dataset.y[test_index]

    # Impute numerical features
    if impute:
        num_idx = np.where(num_mask)[0]

        # The imputer drops columns that are fully NaN. So, we first identify columns that are fully NaN and set them to
        # zero. This will effectively drop the columns without changing the column indexing and ordering that many of
        # the functions in this repository rely upon.
        fully_nan_num_idcs = np.nonzero((~np.isnan(X_train[:, num_idx].astype("float"))).sum(axis=0) == 0)[0]
        if fully_nan_num_idcs.size > 0:
            X_train[:, num_idx[fully_nan_num_idcs]] = 0
            X_val[:, num_idx[fully_nan_num_idcs]] = 0
            X_test[:, num_idx[fully_nan_num_idcs]] = 0

        # Impute numerical features, and pass through the rest
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer())]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_idx),
                ('pass', 'passthrough', dataset.cat_idx),
                #("cat", categorical_transformer, categorical_features),
            ],
            #remainder="passthrough",
        )
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)

        # Re-order columns (ColumnTransformer permutes them)
        perm_idx = []
        running_num_idx = 0
        running_cat_idx = 0
        for is_num in num_mask:
            if is_num > 0:
                perm_idx.append(running_num_idx)
                running_num_idx += 1
            else:
                perm_idx.append(running_cat_idx + len(num_idx))
                running_cat_idx += 1
        assert running_num_idx == len(num_idx)
        assert running_cat_idx == len(dataset.cat_idx)
        X_train = X_train[:, perm_idx]
        X_val = X_val[:, perm_idx]
        X_test = X_test[:, perm_idx]

    if scaler != "None":
        if verbose:
            print(f"Scaling the data using {scaler}...")
        X_train[:, num_mask] = scaler_function.fit_transform(X_train[:, num_mask])
        X_val[:, num_mask] = scaler_function.transform(X_val[:, num_mask])
        X_test[:, num_mask] = scaler_function.transform(X_test[:, num_mask])

    if one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        new_x1 = ohe.fit_transform(X_train[:, dataset.cat_idx])
        X_train = np.concatenate([new_x1, X_train[:, num_mask]], axis=1)
        new_x1_test = ohe.transform(X_test[:, dataset.cat_idx])
        X_test = np.concatenate([new_x1_test, X_test[:, num_mask]], axis=1)
        new_x1_val = ohe.transform(X_val[:, dataset.cat_idx])
        X_val = np.concatenate([new_x1_val, X_val[:, num_mask]], axis=1)
        if verbose:
            print("New Shape:", X_train.shape)

    # create subset of dataset if needed
    if (args.subset_features > 0 or args.subset_rows > 0) and (args.subset_features < args.num_features or args.subset_rows < len(X_train)):
        if getattr(dataset, 'ssm', None) is None:
            dataset.ssm = SubsetMaker(args.subset_features, args.subset_rows, args.subset_features_method, args.subset_rows_method, args.y_equalizer)
        X_train_l = []
        y_train_l = []
        X_val_l = []
        y_val_l = []
        X_test_l = []
        y_test_l = []
        if 0 < args.subset_features < args.num_features:
            X_train_t, y_train_t = dataset.ssm.make_subset(
                X_train,
                y_train,
                X_val,
                split='train',
                num_classes=dataset.num_classes,
                rand_seed=args.hparam_seed,
                action=['features'],
            )
            X_val_t, y_val_t = dataset.ssm.make_subset(
                X_val,
                y_val,
                None,
                split='val',
                num_classes=dataset.num_classes,
                rand_seed=args.hparam_seed,
                action=['features'],
            )
            X_test_t, y_test_t = dataset.ssm.make_subset(
                X_test,
                y_test,
                None,
                split='test',
                num_classes=dataset.num_classes,
                rand_seed=args.hparam_seed,
                action=['features'],
            )
        if 0 < args.subset_rows < len(X_train):
            for j in range(args.num_ensembles):
                print(f"making subset {j} with {args.subset_features} features and {args.subset_rows} rows...")
                X_train_t, y_train_t = dataset.ssm.make_subset(
                    X_train,
                    y_train,
                    X_val,
                    split='train',
                    num_classes=dataset.num_classes,
                    rand_seed=args.hparam_seed + j,
                    action=['rows'],
                )
                X_train_l.append(X_train_t)
                y_train_l.append(y_train_t)
                X_val_l.append(X_val)
                y_val_l.append(y_val)
                X_test_l.append(X_test)
                y_test_l.append(y_test)
        else:
            X_train_l = [X_train]*args.num_ensembles
            y_train_l = [y_train]*args.num_ensembles
        X_val_l = [X_val]*args.num_ensembles
        y_val_l = [y_val]*args.num_ensembles
        X_test_l = [X_test]*args.num_ensembles
        y_test_l = [y_test]*args.num_ensembles

    return {
        "args" : args,
        "data_train": (X_train_l, y_train_l),
        "data_val": (X_val_l, y_val_l),
        "data_test": (X_test_l, y_test_l),
    }
