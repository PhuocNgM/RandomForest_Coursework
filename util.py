import numpy as np

def cross_val_score(estimator, X, y, cv, scoring):
    """
    A simple implementation of cross_val_score to mimic sklearn's behavior.
    This function splits the data into `cv` folds, trains the estimator on
    `cv-1` folds and evaluates on the remaining fold, repeating this process
    for each fold. It returns an array of scores.
    """
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        
        if scoring == 'accuracy':
            accuracy = np.mean(y_pred == y_test)
            scores.append(accuracy)
        else:
            raise ValueError("Unsupported scoring method: {}".format(scoring))
    
    return np.array(scores)