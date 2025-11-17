from math import sqrt
from random import Random
import numpy as np
import pandas as pd
from collections import Counter

# module.py
# Custom Random Forest classifier implementation (API similar to sklearn)
# Uses decision tree logic adapted from fromscratch.py


class CustomRandomForestClassifier:
    """
    Minimal Random Forest classifier with sklearn-like API:
      - init(n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt',
             bootstrap=True, max_samples=1.0, random_state=None)
      - fit(X, y)
      - predict(X)
      - predict_proba(X)
    Works with numpy arrays or pandas DataFrame/Series.
    """
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2,
                 max_features='sqrt', bootstrap=True, max_samples=1.0, random_state=None):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.max_samples = float(max_samples)
        self.random_state = random_state
        self.trees = []
        self._rng = Random(random_state)
        self.n_features_in_ = None
        self.classes_ = None

    # ----------------- utilities to convert input -----------------
    def _to_dataset(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).ravel()
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        dataset = [list(X[i]) + [y[i]] for i in range(X.shape[0])]
        return dataset

    # ----------------- tree helpers (ported & adapted) -----------------
    def _test_split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def _gini_index(self, groups, classes):
        n_instances = float(sum(len(g) for g in groups))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def _to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return Counter(outcomes).most_common(1)[0][0]

    def _get_split(self, dataset, n_features):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = None, None, float('inf'), None
        features = []
        while len(features) < n_features:
            idx = self._rng.randrange(len(dataset[0]) - 1)
            if idx not in features:
                features.append(idx)
        for index in features:
            for row in dataset:
                groups = self._test_split(index, row[index], dataset)
                gini = self._gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def _split(self, node, depth, n_features):
        left, right = node['groups']
        del node['groups']
        if not left or not right:
            node['left'] = node['right'] = self._to_terminal(left + right)
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self._to_terminal(left), self._to_terminal(right)
            return
        if len(left) <= self.min_samples_split:
            node['left'] = self._to_terminal(left)
        else:
            node['left'] = self._get_split(left, n_features)
            self._split(node['left'], depth + 1, n_features)
        if len(right) <= self.min_samples_split:
            node['right'] = self._to_terminal(right)
        else:
            node['right'] = self._get_split(right, n_features)
            self._split(node['right'], depth + 1, n_features)

    def _build_tree(self, train, n_features):
        root = self._get_split(train, n_features)
        self._split(root, 1, n_features)
        return root

    def _predict_tree(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict_tree(node['left'], row)
            return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict_tree(node['right'], row)
            return node['right']

    # ----------------- sampling -----------------
    def _subsample(self, dataset, ratio):
        n_sample = round(len(dataset) * ratio)
        sample = [dataset[self._rng.randrange(len(dataset))] for _ in range(n_sample)]
        return sample

    # ----------------- public API -----------------
    def fit(self, X, y):
        dataset = self._to_dataset(X, y)
        self.n_features_in_ = (len(dataset[0]) - 1) if dataset else 0
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                n_features = max(1, int(sqrt(self.n_features_in_)))
            elif self.max_features == 'log2':
                n_features = max(1, int(np.log2(self.n_features_in_)))
            else:
                n_features = self.n_features_in_
        else:
            n_features = int(self.max_features)
        self.classes_ = sorted(list(set([row[-1] for row in dataset])))
        self.trees = []
        for i in range(self.n_estimators):
            if self.bootstrap:
                sample = self._subsample(dataset, self.max_samples)
            else:
                sample = list(dataset)
            tree = self._build_tree(sample, n_features)
            self.trees.append(tree)
        return self

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = np.asarray(X)
        X = np.asarray(X)
        preds = []
        for r in range(X.shape[0]):
            row = list(X[r])
            votes = [self._predict_tree(t, row) for t in self.trees]
            pred = Counter(votes).most_common(1)[0][0]
            preds.append(pred)
        return np.array(preds)

    def predict_proba(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = np.asarray(X)
        X = np.asarray(X)
        proba = []
        for r in range(X.shape[0]):
            row = list(X[r])
            votes = [self._predict_tree(t, row) for t in self.trees]
            counts = Counter(votes)
            total = sum(counts.values())
            probs = [counts.get(c, 0) / total for c in self.classes_]
            proba.append(probs)
        return np.asarray(proba)

    # convenience score (accuracy)
    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.asarray(y).ravel()
        return np.mean(y_pred == y_true)