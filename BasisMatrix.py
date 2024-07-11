import numpy as np
import matplotlib.pyplot as plt


class BasisMatrix:
    """
    BasisMatrix represents the Basis vector matrix.
    """

    def __init__(self):
        """
        BasisMatrix constructor.
        """

    def basis_chip(self, u, v, K):
        """
        Generate basis chip based on spatial frequency pair (u, v)

        @param u is the first frequency number
        @param v is the second frequency number
        @param K is the chip size
        @return basis chip matrix
        """

        # create an empty KxK chip
        chip = np.zeros((K, K))

        # normalization
        alpha_u = np.sqrt(1 / K) if u == 1 else np.sqrt(2 / K)
        beta_v = np.sqrt(1 / K) if v == 1 else np.sqrt(2 / K)

        # dct coefficients
        for x in range(1, K + 1):
            for y in range(1, K + 1):
                chip[x - 1, y - 1] = (
                    alpha_u
                    * beta_v
                    * np.cos((2 * y - 1) * (u - 1) * np.pi / (2 * K))
                    * np.cos((2 * x - 1) * (v - 1) * np.pi / (2 * K))
                )
        return chip

    def basis_vector_matrix(self, K):
        """
        Generate Basis vector matrix.

        @param K is the chip size
        @return basis chip matrix of size K^2 * K^2
        """

        # init matrix
        vector_matrix = np.zeros((K**2, K**2))

        # calculate basis vector matrix
        for v in range(1, K + 1):
            for u in range(1, K + 1):
                # rasterize
                vector = self.basis_chip(u, v, K).flatten()
                vector.reshape((K**2, 1))

                col = (v - 1) * K + (u - 1)
                vector_matrix[:, col] = vector

        return vector_matrix


if __name__ == "__main__":
    # create BasisMatrix, basis chip and basis matrix
    u = 8
    v = 4
    K = 16
    basis = BasisMatrix()
    basis_chip = basis.basis_chip(u, v, K)
    basis_matrix = basis.basis_vector_matrix(K)

    # display basis chip
    # plt.imshow(basis_chip, cmap="gray")
    # plt.title(f"Basis Chip T(U={u}, V={v})")
    # plt.xlabel("Sample Index (u)")
    # plt.ylabel("Sample Index (v)")
    # plt.show()

    # display basis vector matrix
    plt.imshow(basis_matrix, cmap="gray")
    plt.title(f"Basis Vector Matrix with U-V Ordering (K={K})")
    plt.xlabel("Basis Vector Index (u,v)")
    plt.ylabel("Pixel Location Index (x,y)")
    plt.colorbar()
    plt.savefig(f"results/basis_vector_matrix_{K}.png", dpi=300)
