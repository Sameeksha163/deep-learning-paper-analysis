import numpy as np

class DeepLasso:
    def __init__(self):
        self.feature_importance_ = None

    def calculate_importance(self, X, model):
        """
        Simulate feature importance calculation using gradients.
        Replace this placeholder with real gradient-based logic.
        """
        # Example: Assign random importance scores
        self.feature_importance_ = np.random.rand(X.shape[1])

    def select(self, X, topk=None):
        """
        Select the top-k features based on importance scores.
        """
        if self.feature_importance_ is None:
            raise ValueError("Feature importance not calculated. Call calculate_importance() first.")

        # Rank features by importance
        sorted_indices = np.argsort(self.feature_importance_)[::-1]
        n_top = int(topk * len(sorted_indices))
        selected_indices = sorted_indices[:n_top]
        print(f"Selected features: {selected_indices}")

        # Return reduced feature set
        return X.iloc[:, selected_indices]
