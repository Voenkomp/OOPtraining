import pandas as pd
import numpy as np



class MyKNNReg:
    def __init__(self, k: int = 3, metric: str='euclidean', weight: str='uniform'):
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight

    def __repr__(self):
        return f"MyKNNReg class: k={self.k}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape

    def _euclidean(self, row: pd.Series) -> pd.Series:
        return np.sqrt(((row - self.X_train) ** 2).sum(axis=1))
    
    def _chebyshev(self, row: pd.Series) -> pd.Series:
        return (np.abs(row - self.X_train)).max(axis=1)
    
    def _manhattan(self, row: pd.Series) -> pd.Series:
        return np.abs(row - self.X_train).sum(axis=1)
    
    def _cosine(self, row: pd.Series) -> pd.Series:
        numerator = (row * self.X_train).sum(axis=1)
        denominator = np.sqrt(np.sum((row**2))) * np.sqrt((self.X_train**2).sum(axis=1))
        return 1 - (numerator / denominator)

    def get_metrics(self, row):
        dist_min_index = getattr(self, '_' + self.metric)(row).sort_values().head(self.k).index
        if self.weight == 'rank':
            
        return self.y_train[dist_min_index].mean()

    def predict(self, X: pd.DataFrame):
        return X.apply(self.get_metrics, axis=1)