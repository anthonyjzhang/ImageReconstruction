from sklearn.linear_model import Lasso


class LassoModel:
    """
    Lasso Model represents a Linear regression model with L1 regularization.
    """

    def __init__(self, alpha=1.0):
        """
        LassoModel constructor.

        @param alpha is the regularization strength.
        """

        self.lasso = Lasso(alpha=alpha)

    def fit(self, A, D):
        """
        Fit the model with input matrix A with target D.

        @param A is the input feature matrix
        @param D is the target vector
        """

        if D.ndim == 1:
            D = D.reshape(-1, 1)

        self.lasso.fit(A, D)

    def predict(self, A):
        """
        Predict on new data A.

        @param A is new data input
        @return prediction vector
        """

        return self.lasso.predict(A)

    def get_coefficients(self):
        """
        Get sparse coefficient matrix.

        @return model coefficients
        """

        return self.lasso.coef_
