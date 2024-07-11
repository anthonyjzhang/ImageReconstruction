# Compressed Sensing Image Reconstruction

## Background
The ability to produce high-quality signal reconstructions from noisy, corrupted, or incomplete data sources is a significant breakthrough in bridging the gap between humans and digital information. Accurate image reconstruction is essential for improving decision-making across various applications, including medical imaging, security surveillance, digital forensics, and content creation. For instance, in medical imaging, reconstruction methods enhance imaging performance and improve diagnosis. This project aims to explore image reconstruction using compressed sensing, a technique that leverages sparsity assumptions to recover signals from limited measurements.

## Project Overview
This project implements a compressed sensing image reconstruction algorithm to predict and fill in missing pixels in corrupted images. Utilizing Discrete Cosine Transform (DCT) and L1-Regularization (Lasso), the algorithm reconstructs high-quality images from significantly corrupted sources. The primary goal of this project is to develop a predictive model for image reconstruction by creating a basis function representation of image chips, training a regression model with supervised learning, and applying L1-Regularization to enforce sparsity. We aim to estimate missing pixels by using the model, optimize the regularization parameter through cross-validation, and evaluate the impact of median filtering on reconstruction quality.

## Methodology
The project begins by representing images using the Discrete Cosine Transform (DCT), which allows us to express images as a weighted sum of sinusoids. This representation is then rasterized into a basis vector matrix for efficient computation. Compressed sensing principles are applied by assuming sparsity in the DCT coefficients, enabling us to reconstruct image blocks accurately. Lasso regression is employed to enforce sparsity in the regression model, solving the underdetermined linear system by optimizing the sparse weight vector. Extensive cross-validation with 20 folds is performed to find the optimal regularization strength, splitting pixels into training and testing sets to minimize mean squared error (MSE) for accurate reconstruction. Finally, median filtering is applied to enhance the quality of reconstructed images by smoothing out impulsive noise.

## Results
The algorithm successfully reconstructed images with up to 65% missing pixels, achieving significant improvements in image quality. Extensive cross-validation and parameter tuning resulted in a mean squared error reduction of 30% in final reconstructions. Reconstructed images demonstrated up to 85% similarity to the original images, highlighting the effectiveness of the approach.
    ```

## References
- Yaqub M, Jinchao F, Arshid K, Ahmed S, Zhang W, Nawaz MZ, Mahmood T. Deep Learning-Based Image Reconstruction for Different Medical Imaging Modalities. Comput Math Methods Med. 2022.
- The MathWorks Inc. Discrete Cosine Transform Documentation, Natick, Massachusetts: The MathWorks Inc.
- F.H.P. Fitzek, F. Granelli, P. Seeling (Eds.), Computing in Communication Networks, Academic Press, 2020.
- Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, pp. 2825-2830, 2011.
- Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362, 2020.
- Hunter, J. D. Matplotlib: A 2D Graphics Environment, Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
