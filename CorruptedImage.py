import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class CorruptedImage:
    """
    CorruptedImage represents a corrupted/test image.
    """

    def __init__(self, file_path, isTest=False):
        """
        CorruptedImage constructor.

        @param file_path is file path of image
        @param isTest is whether image is test image or not
        """

        if file_path.endswith(".txt"):
            self.image = np.genfromtxt(file_path, delimiter=",")
        else:
            self.image = mpimg.imread(file_path)

        self.image = np.where(
            np.isnan(self.image), -np.inf, self.image
        )  # replace Nan with -inf
        self.isTest = isTest
        self.shape = self.image.shape

    def chip(self, x, y, K, S=None):
        """
        Get the KxK chip from position x, y.

        @param x is the x position of chip
        @param y is the y position of chip
        @param K is the chip size
        @param S is the number of sensed pixels (only valid if image is test image)
        """

        x_star = K * (x)
        y_star = K * (y)
        block = self.image[y_star : (y_star + K), x_star : (x_star + K)]

        if self.isTest and S != None:
            # mask for sensed pixels
            mask = np.zeros((K, K), dtype=bool)
            sensed_indices = np.random.choice(K * K, S, replace=False)
            mask.ravel()[sensed_indices] = True

            # missing values set to -inf
            sensed_matrix = np.full((K, K), -np.inf)
            sensed_matrix[mask] = block[mask]

            return sensed_matrix

        return block

    def corrupt_test_image(self, K, S):
        rows, cols = self.shape
        corrupted = np.full_like(self.image, -np.inf)
        for x in range((cols + K - 1) // K):
            for y in range((rows + K - 1) // K):
                # get bounds of chip
                x_start = K * x
                y_start = K * y
                x_end = min(x_start + K, cols)
                y_end = min(y_start + K, rows)

                # get corrupted chip
                corrupted_chip = self.chip(x, y, K, S)

                corrupted[y_start:y_end, x_start:x_end] = corrupted_chip[
                    : (y_end - y_start), : (x_end - x_start)
                ]

        return corrupted;

    def display_image(self):
        """
        Display image.
        """

        plt.imshow(self.image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    # create CorruptedImage from nature image
    file_path = "data/nature.bmp"
    img = CorruptedImage(file_path, isTest=True)

    # img.display_image()

    # define chip location, chip size and sensing
    x, y = 0, 0
    K = 16
    S = 30
    S_values = [10, 30, 50, 100, 150]

    # display original from test image
    # chip = img.chip(x, y, K)
    # plt.imshow(chip, cmap="gray")
    # plt.title(f"{K}x{K} Image Chip at ({K*(x)}, {K*(y)})")
    # plt.axis("on")
    # plt.show()

    # # display sensed chip from test image
    # sensed_chip = img.chip(x, y, K, S)
    # fig, ax = plt.subplots()
    # cmap = plt.cm.gray
    # cmap.set_bad(color="orange")
    # ax.imshow(np.ma.masked_where(sensed_chip == -np.inf, sensed_chip), cmap=cmap)
    # ax.set_title(f"{K}x{K} Image Chip with {S} sensed pixels")
    # ax.axis("on")
    # plt.show()

    # display original image with corrupted images of different S
    plt.imshow(img.image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    for i, S in enumerate(S_values, start=1):
        corrupted = img.corrupt_test_image(K=K, S=S)

        plt.imshow(corrupted, cmap="gray", interpolation="nearest")
        plt.title(f"Corrupted Image S={S}")
        plt.axis("off")
        plt.show()
