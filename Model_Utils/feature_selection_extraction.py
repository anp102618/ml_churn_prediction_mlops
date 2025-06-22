from abc import ABC, abstractmethod
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from typing import Optional, List


# -------------------- Abstract Base --------------------

class FeatureProcessor(ABC):
    """
    Abstract base class for feature selection or extraction strategies.
    All subclasses must implement fit() and transform().
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


# -------------------- Feature Selection --------------------

class SelectKBestProcessor(FeatureProcessor):
    """
    Feature selection using SelectKBest with f_classif (ANOVA F-test).
    """

    def __init__(self, k: int = 5):
        self.selector = SelectKBest(score_func=f_classif, k=k)
        self.selected_columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        if y is None:
            raise ValueError("SelectKBest requires target variable y.")
        if X.empty or y.empty:
            raise ValueError("Input data X or y is empty.")
        self.selector.fit(X, y.values.ravel())
        self.selected_columns = X.columns[self.selector.get_support()].tolist()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            raise ValueError("Call fit() before transform().")
        return pd.DataFrame(self.selector.transform(X[self.selected_columns]),
                            columns=self.selected_columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


class VarianceThresholdProcessor(FeatureProcessor):
    """
    Feature selection by removing low-variance features.
    """

    def __init__(self, threshold: float = 0.0):
        self.selector = VarianceThreshold(threshold)
        self.selected_columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        if X.empty:
            raise ValueError("Input data X is empty.")
        self.selector.fit(X)
        self.selected_columns = X.columns[self.selector.get_support()].tolist()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            raise ValueError("Call fit() before transform().")
        return pd.DataFrame(self.selector.transform(X[self.selected_columns]),
                            columns=self.selected_columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


# -------------------- Feature Extraction --------------------

class LDAProcessor(FeatureProcessor):
    """
    Linear Discriminant Analysis (LDA) for supervised feature extraction.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.lda = LDA(n_components=n_components)
        self.columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        if y is None:
            raise ValueError("LDA requires target variable y.")
        if X.empty or y.empty:
            raise ValueError("Input data X or y is empty.")
        X_new = self.lda.fit_transform(X, y.values.ravel())
        self.columns = [f"LDA{i + 1}" for i in range(X_new.shape[1])]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            raise ValueError("Call fit() before transform().")
        X_new = self.lda.transform(X)
        return pd.DataFrame(X_new, columns=self.columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


class KernelPCAProcessor(FeatureProcessor):
    """
    Kernel PCA for nonlinear dimensionality reduction.
    """

    def __init__(self, n_components: int = 10, kernel: str = 'rbf'):
        self.kpca = KernelPCA(n_components=n_components, kernel=kernel, random_state=42)
        self.columns: Optional[List[str]] = None
        self.input_columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        if X.empty:
            raise ValueError("Input data X is empty.")
        self.input_columns = X.columns.tolist()
        X_new = self.kpca.fit_transform(X)
        self.columns = [f"KPC{i + 1}" for i in range(X_new.shape[1])]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns or not self.input_columns:
            raise ValueError("Call fit() before transform().")
        X = X[self.input_columns]
        X_new = self.kpca.transform(X)
        return pd.DataFrame(X_new, columns=self.columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


# -------------------- Feature Factory --------------------

class FeatureFactory:
    """
    Factory for creating feature processors.

    Usage:
        FeatureFactory.get_processor(kind="selection", method="selectkbest", k=10)
    """

    @staticmethod
    def get_processor(kind: str, method: str, **kwargs) -> FeatureProcessor:
        kind = kind.lower()
        method = method.lower()

        if kind == 'selection':
            if method == 'selectkbest':
                return SelectKBestProcessor(**kwargs)
            elif method == 'variancethreshold':
                return VarianceThresholdProcessor(**kwargs)

        elif kind == 'extraction':
            if method == 'lda':
                return LDAProcessor(**kwargs)
            elif method == 'kernelpca':
                return KernelPCAProcessor(**kwargs)

        raise ValueError(f"Unknown kind '{kind}' or method '{method}'.")
