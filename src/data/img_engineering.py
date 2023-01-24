import numpy as np
import cv2


def get_first_n_pcs(img: np.array, num_components: int):
    """Perform PCA on a single image and return principle components which make up the most variance.

    Args:
        img (np.array): Original image of shape (h, w, num_bands).
        num_components (int): Desired number of components to be returned.

    Returns:
        np.array: Components with shape (h, w, num_components).
    """
    # Convert 2d bands into 1-d arrays
    bands_vectorized = np.zeros(shape=(img.shape[0] * img.shape[1], img.shape[2]))
    for i in range(img.shape[-1]):
        flattened_band = img[:, :, i].flatten()
        flattened_band_standard = (
            flattened_band - flattened_band.mean()
        ) / flattened_band.std()
        bands_vectorized[:, i] = flattened_band_standard

    cov = np.cov(bands_vectorized.transpose())
    eig_val, eig_vec = np.linalg.eig(cov)

    # Ordering Eigen values and vectors
    order = eig_val.argsort()[::-1]
    eig_val = eig_val[order]
    eig_vec = eig_vec[:, order]

    # Projecting data on Eigen vector directions resulting in Principal Components
    pcs = np.matmul(bands_vectorized, eig_vec)

    # Rearranging 1-d arrays to 2-d arrays of image size
    PC_2d = np.zeros((img.shape[0], img.shape[1], num_components))
    for i in range(num_components):
        PC_2d[:, :, i] = pcs[:, i].reshape(-1, img.shape[1])

    # normalizing between 0 to 255
    PC_2d_Norm = np.zeros((img.shape[0], img.shape[1], num_components))
    for i in range(num_components):
        PC_2d_Norm[:, :, i] = cv2.normalize(
            PC_2d[:, :, i], np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX
        )

    return PC_2d_Norm[:, :, : num_components + 1]
