from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class FeatureFactory:
    """
    Factory for returning sklearn feature selection/extraction instances.

    Args:
        kind (str): Either 'selection' or 'extraction'.
        method (str): Method name like 'selectkbest', 'lda', etc.
        kwargs: Parameters to pass to the sklearn constructor.

    Returns:
        sklearn object (e.g., SelectKBest, LDA)
    """

    @staticmethod
    def get_processor(kind: str, method: str, **kwargs):
        kind = kind.lower()
        method = method.lower()

        if kind == "selection":
            if method == "selectkbest":
                return SelectKBest(score_func=f_classif, **kwargs)
            elif method == "variancethreshold":
                return VarianceThreshold(**kwargs)

        elif kind == "extraction":
            if method == "lda":
                return LDA(**kwargs)
            elif method == "kernelpca":
                return KernelPCA(**kwargs)

        raise ValueError(f"Invalid kind='{kind}' or method='{method}'.")
