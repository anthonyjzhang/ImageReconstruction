import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from LassoModel import LassoModel
from CorruptedImage import CorruptedImage


class CrossValidation:
    """
    CrossValidation performs a L-k fold cross validation split on a chip level.
    """

    def __init__(self, n_splits=20):
        """
        CrossValidation constructor.

        @param n_splits defines how many subsets to use during cross validation.
        """

        self.n_splits = n_splits
        self.regularization_values = [
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            0.01,
            0.04,
            0.10,
            0.15,
            0.20,
            0.3,
            0.5,
            0.8,
            1e1,
            1e2,
            1e3,
            1e4,
            1e5,
            1e6
        ]

    def find_optimal_alpha(self, corrupted_chip, basis_matrix):
        """
        Find the optimal alpha through cross validation with random subsets.

        @param corrupted_chip is the chip with missing pixels
        @param basis_matrix is the basis vector matrix
        """

        average_mse_values = self.cross_validation(corrupted_chip, basis_matrix)
        optimal_lambda_index = np.argmin(average_mse_values)
        optimal_lambda = self.regularization_values[optimal_lambda_index]
        return optimal_lambda

    def cross_validation(self, corrupted_chip, basis_matrix):
        """
        Perform M iterations of cross validation with random subsets on corrupted chip.

        @param corrupted_chip is the chip with missing pixels.
        @basis_matrix is the basis vector matrix
        """

        # loop through all splits and calculate mse for all regularization values for each split
        sensed = np.sum(corrupted_chip != -np.inf)
        all_mse_values = np.zeros((self.n_splits, len(self.regularization_values)))
        for split in range(self.n_splits):
            # calculate m for the test set
            m = sensed // 6

            # get indices of the sensed pixels
            sensed_indices = np.where(np.isfinite(corrupted_chip.flatten()))[0]

            # randomly select m pixels for the test set and the rest for the training set
            test_indices = np.random.choice(sensed_indices, m, replace=False)
            train_indices = np.setdiff1d(sensed_indices, test_indices)

            # loop through each regularization value
            for i, alpha in enumerate(self.regularization_values):
                # create new model with alpha
                model = LassoModel(alpha=alpha)

                # fit model with training indices of basis matrix and sensed corruped chip
                model.fit(
                    basis_matrix[train_indices, :],
                    corrupted_chip.flatten()[train_indices],
                )

                # predict reconstructed test pixels
                reconstructed_test_pixels = model.predict(basis_matrix[test_indices, :])

                # calculate mean squared error
                all_mse_values[split, i] = mean_squared_error(
                    corrupted_chip.flatten()[test_indices], reconstructed_test_pixels
                )

        # calculate average MSE across all splits for each alpha
        average_mse_values = np.mean(all_mse_values, axis=0)
        return average_mse_values


if __name__ == "__main__":
    file_path = "field_test_image.txt"
    image = CorruptedImage(file_path)
    cv = CrossValidation()

    
    
