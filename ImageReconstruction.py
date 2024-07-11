import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import median_filter
from sklearn.metrics import mean_squared_error
from BasisMatrix import BasisMatrix
from CrossValidation import CrossValidation
from CorruptedImage import CorruptedImage
from LassoModel import LassoModel

warnings.filterwarnings("ignore")


class ImageReconstruction:
    """
    ImageReconstruction is the interface to reconstruct a corrupted image.
    """

    def __init__(self, file_path, isTest=False):
        """
        ImageReconstruction constructor.

        @param file_path is the file path of the corrupted/test image
        @param isTest is a boolean indicating if the image is a test image or not
        """

        self.image = CorruptedImage(file_path=file_path, isTest=isTest)
        self.basis = BasisMatrix()
        self.cv = CrossValidation()
        self.reconstructed = np.zeros_like(self.image.image)
        self.reconstructed_filtered = np.zeros_like(self.image.image)
        self.isTest = isTest
        self.alphas = []
        if isTest:
            self.corrupted = np.full_like(self.image.image, -np.inf)

    def reconstruct_image(self, K, S=None):
        """
        Reconstructs the whole image by processing each KxK block

        @param K is the chip size
        @param S is the sensing (only valid if image is test image)
        """

        if self.isTest and S is None:
            raise ValueError("Test Image requires S value.")

        # generate basis matrix
        basis_matrix = self.basis.basis_vector_matrix(K=K)

        # loop over all blocks in the image
        rows, cols = self.image.shape

        # store alphas
        alpha_map = np.zeros((rows // K, cols // K))

        for x in tqdm(range((cols + K - 1) // K), desc="Reconstructing image"):
            for y in range((rows + K - 1) // K):
                # get bounds of chip
                x_start = K * x
                y_start = K * y
                x_end = min(x_start + K, cols)
                y_end = min(y_start + K, rows)

                # get corrupted chip
                corrupted_chip = self.image.chip(x, y, K, S)

                # get optimal lambda using cross validation
                optimal_alpha = self.cv.find_optimal_alpha(corrupted_chip, basis_matrix)

                # store in alpha map
                self.alphas.append(optimal_alpha)
                alpha_map[y, x] = optimal_alpha

                # reconstruct chip with Lasso model and optimal alpha
                reconstructed_chip = self.reconstruct_chip(
                    corrupted_chip, basis_matrix, optimal_alpha, K
                )

                # place reconstructed chip in image
                self.reconstructed[y_start:y_end, x_start:x_end] = reconstructed_chip[
                    : (y_end - y_start), : (x_end - x_start)
                ]

                # place corrupted chip in corrupted test image
                if self.isTest:
                    self.corrupted[y_start:y_end, x_start:x_end] = corrupted_chip[
                        : (y_end - y_start), : (x_end - x_start)
                    ]

        # apply median filtering to the reconstructed image
        self.reconstructed_filtered = median_filter(self.reconstructed, size=3)

        return self.reconstructed_filtered, alpha_map
    
    def visualize_alpha_map(self, alpha_map):
        """
        Visualizes the log10 of the regularization parameter across the image.
        Each block corresponds to a pixel in the visualization, with log10(alpha) as the pixel intensity.
        """

        log_alpha_map = np.log10(alpha_map)
        plt.imshow(log_alpha_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='log10(Regularization Parameter)')
        plt.title('Variance of Regularization Parameter across the Image, S=30')
        plt.show()

    def reconstruct_chip(self, corrupted_chip, basis_matrix, alpha, K):
        """
        Reconstructs a KxK chip given a basis matrix and an alpha

        @param corrupted_chip is the chip with missing pixels
        @param basis_matrix is the basis vector matrix
        @param alpha is the regularization strength
        @param K is the chip size
        """

        model = LassoModel(alpha=alpha)
        flattened_chip = corrupted_chip.flatten()
        sensed_indices = np.where(np.isfinite(flattened_chip))[0]

        # fit the model on sensed pixels
        model.fit(basis_matrix[sensed_indices, :], flattened_chip[sensed_indices])

        # ifentify unsensed (missing) pixel indices correctly
        unsensed_indices = np.where(np.isneginf(flattened_chip))[0]

        # predict values only for unsensed pixels
        if len(unsensed_indices) > 0:  # ensure there are unsensed pixels to predict
            predicted_pixels = model.predict(basis_matrix[unsensed_indices, :])
            flattened_chip[unsensed_indices] = predicted_pixels

        reconstructed_chip = flattened_chip.reshape(K, K)
        return reconstructed_chip


if __name__ == "__main__":
    # create ImageReconstruction instance and reconstruct image
    file_path = "data/field_test_image.txt"
    file_path = "data/nature.bmp"
    imageReconstruction = ImageReconstruction(file_path=file_path, isTest=True)
    reconstructed = imageReconstruction.reconstruct_image(K=8, S=30)

    # display reconstructed image
    plt.imshow(reconstructed, cmap="gray")
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.show()

    # save the reconstructed matrix to a file
    np.savetxt("reconstructed_image.txt", reconstructed, fmt="%f")

    file_path = "data/nature.bmp"
    S_values = [150]
    K = 16

    for S in S_values:
        imageReconstruction = ImageReconstruction(file_path=file_path, isTest=True)
        
        original = imageReconstruction.image.image
        reconstructed_filtered = imageReconstruction.reconstruct_image(K=K, S=S)
        reconstructed_no_filter = imageReconstruction.reconstructed
        corrupted = imageReconstruction.corrupted
        
        # Calculate MSE
        mse_no_filter = mean_squared_error(original, reconstructed_no_filter)
        mse_filtered = mean_squared_error(original, reconstructed_filtered)
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axs[0].imshow(original, cmap='gray')
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        
        # Corrupted image
        axs[1].imshow(corrupted, cmap='gray', interpolation='nearest')
        axs[1].set_title(f'Corrupted Image, S={S}')
        axs[1].axis('off')
        
        # Reconstructed without filtering
        axs[2].imshow(reconstructed_no_filter, cmap='gray')
        axs[2].set_title(f'Reconstructed without filter\nMSE: {mse_no_filter:.2f}')
        axs[2].axis('off')
        
        # Reconstructed with median filtering
        axs[3].imshow(reconstructed_filtered, cmap='gray')
        axs[3].set_title(f'Reconstructed with filter\nMSE: {mse_filtered:.2f}')
        axs[3].axis('off')
        
        plt.show()

    file_path = "data/nature.bmp"
    imageReconstruction = ImageReconstruction(file_path=file_path, isTest=True)
    reconstructed, alpha_map = imageReconstruction.reconstruct_image(K=16, S=50)
    imageReconstruction.visualize_alpha_map(alpha_map)