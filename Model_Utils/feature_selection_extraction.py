from abc import ABC, abstractmethod
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from typing import Optional, Union, List, Dict


class FeatureProcessor(ABC):
    """
    Abstract base class for feature selection or extraction strategies.
    """
    @abstractmethod
    def process(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Apply the feature processing strategy.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (Optional[pd.Series]): Target variable if required.

        Returns:
            pd.DataFrame: Transformed feature matrix.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature processor.

        Args:
            X (pd.DataFrame): The input feature matrix to transform.

        Returns:
            pd.DataFrame: The transformed feature matrix.
        """
        pass
# ========================== Selection Strategies ===========================

class SelectKBestStrategy(FeatureProcessor):
    def __init__(self, k: int = 5):
        self.k = k
        self.selector = SelectKBest(score_func=f_classif, k=self.k)
        self.selected_columns = None
        self.is_fitted = False

    def process(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if y is None:
            raise ValueError("Target variable 'y' is required for SelectKBest.")
        if X.empty:
            raise ValueError("Input DataFrame 'X' is empty.")
        
        X_new = self.selector.fit_transform(X, y)
        self.selected_columns = X.columns[self.selector.get_support()].tolist()
        self.is_fitted = True
        
        return pd.DataFrame(X_new, columns=self.selected_columns, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted or self.selected_columns is None:
            raise ValueError("You must call process() before transform().")
        
        X_selected = X[self.selected_columns]
        X_new = self.selector.transform(X_selected)
        
        return pd.DataFrame(X_new, columns=self.selected_columns, index=X.index)

    def get_support(self, indices: bool = False):
        if not self.is_fitted:
            raise ValueError("Call process() before get_support().")
        return self.selector.get_support(indices=indices)

    def get_selected_columns(self) -> Optional[list]:
        return self.selected_columns


class VarianceThresholdStrategy(FeatureProcessor):
    """
    Removes features with variance below a certain threshold.
    """
    def __init__(self, threshold: float = 0.0):
        self.selector = VarianceThreshold(threshold=threshold)
        self.selected_columns = None
        self.is_fitted = False

    def process(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if X.empty:
            raise ValueError("Input DataFrame 'X' is empty.")
        X_new = self.selector.fit_transform(X)
        self.selected_columns = X.columns[self.selector.get_support()]
        self.is_fitted = True
        return pd.DataFrame(X_new, columns=self.selected_columns, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("You must call `process()` before `transform()`.")
        if X.empty:
            raise ValueError("Input DataFrame 'X' is empty.")
        X_new = self.selector.transform(X)
        return pd.DataFrame(X_new, columns=self.selected_columns, index=X.index)

# ========================== Extraction Strategies ===========================

class LDAStrategy(FeatureProcessor):
    """
    Applies Linear Discriminant Analysis for dimensionality reduction.
    """
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.lda = LDA(n_components=self.n_components)
        self.columns = None
        self.is_fitted = False

    def process(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if y is None:
            raise ValueError("Target variable 'y' is required for LDA.")
        if X.empty:
            raise ValueError("Input DataFrame 'X' is empty.")
        X_new = self.lda.fit_transform(X, y)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        self.columns = [f'LDA{i+1}' for i in range(X_new.shape[1])]
        self.is_fitted = True
        return pd.DataFrame(X_new, columns=self.columns, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("You must call `process()` before `transform()`.")
        if X.empty:
            raise ValueError("Input DataFrame 'X' is empty.")
        X_new = self.lda.transform(X)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        return pd.DataFrame(X_new, columns=self.columns, index=X.index)

    
class KernelPCAStrategy(FeatureProcessor):
    """
    Applies Kernel PCA transformation for nonlinear feature extraction.
    """
    def __init__(self, n_components: int = 15, kernel: str = 'rbf'):
        self.n_components = n_components
        self.kernel = kernel
        self.kpca = KernelPCA(n_components=self.n_components, kernel=self.kernel, random_state=42)
        self.input_columns: Optional[List[str]] = None
        self.component_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def process(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if X.empty:
            raise ValueError("Input DataFrame 'X' is empty.")
        
        self.input_columns = X.columns.tolist()
        X_new = self.kpca.fit_transform(X)
        self.component_names = [f'KPC{i+1}' for i in range(self.n_components)]
        self.is_fitted = True

        return pd.DataFrame(X_new, columns=self.component_names, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("You must call process() before transform().")
        
        X = X[self.input_columns]
        X_new = self.kpca.transform(X)
        
        return pd.DataFrame(X_new, columns=self.component_names, index=X.index)

    def get_components(self) -> Optional[List[str]]:
        return self.component_names


# ============================ Strategy Factory ============================

class FeatureFactory:
    """
    Factory class for creating feature processors for selection or extraction.
    """
    @staticmethod
    def get_processor(kind: str, method: str, **kwargs) -> FeatureProcessor:
        """
        Instantiates the requested feature processor.

        Args:
            kind (str): Either 'selection' or 'extraction'.
            method (str): Name of the method to apply.
            kwargs: Additional parameters for the processor.

        Returns:
            FeatureProcessor: An instance of a feature processor.

        Raises:
            ValueError: If an unsupported kind or method is specified.
        """
        kind = kind.lower()
        method = method.lower()

        if kind == 'selection':
            if method == 'selectkbest':
                return SelectKBestStrategy(**kwargs)
            elif method == 'variancethreshold':
                return VarianceThresholdStrategy(**kwargs)
            else:
                raise ValueError(f"Unknown selection method '{method}'")
        
        elif kind == 'extraction':
            if method == 'lda':
                return LDAStrategy(**kwargs)
            elif method == 'kernelpca':
                return KernelPCAStrategy(**kwargs)
            else:
                raise ValueError(f"Unknown extraction method '{method}'")
        
        else:
            raise ValueError(f"Unknown feature processor kind '{kind}'")
