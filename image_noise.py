import random
import ssl
import urllib.request
from typing import Literal, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image


# Read file as ndarray
def load_image(filename: str):
    image = Image.open(filename)
    return np.asarray(image)


# Save ndarray as file
def save_image(image: NDArray, filename: str):
    if image.dtype != np.uint8:
        image = (255 * (image - image.min()) / (image.max() - image.min())).astype(
            np.uint8
        )  # convert to uint8 if needeed (gaussian and speckle)

    image_data = Image.fromarray(image)
    image_data.save(filename)


# Add gaussian, salt & pepper, or speckle noise to an image
def add_noise(
    noise_type: Union[
        Literal["gaussian"], Literal["salt & pepper"], Literal["speckle"]
    ],
    image: np.ndarray,
):
    row, col, ch = image.shape

    if noise_type == "gaussian":
        # Gaussian parameters
        mean = 0
        sigma = 15

        # Math
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss

        return noisy
    elif noise_type == "salt & pepper":
        # Parameters
        threshold = 0.005  # half the proportion of pixels affected
        high = 250  # salt
        low = 5  # pepper

        # Math
        noisy = np.copy(image)
        random_matrix = np.random.rand(row, col)
        noisy[random_matrix >= (1 - threshold)] = high
        noisy[random_matrix <= threshold] = low

        return noisy
    elif noise_type == "speckle":
        # Parameters
        variance = 0.25  # strength of the multiplicative noise

        # Math
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss * variance

        return noisy


def standard_denoise(image: NDArray):
    if image.dtype != np.uint8:
        image = (255 * (image - image.min()) / (image.max() - image.min())).astype(
            np.uint8
        )  # convert to uint8 if needeed (gaussian and speckle)
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


# For each image URL in ./image.txt (separated by newlines), download it and apply a random noise type
# Files are saved in ./images/
if __name__ == "__main__":
    ssl._create_default_https_context = (
        ssl._create_unverified_context
    )  # needed to avoid SSL errors

    with open("images.txt", "r") as images:
        for index, image in enumerate(images):
            i = index + 1

            urllib.request.urlretrieve(image, f"images/{i}-clean.jpg")

            image = load_image(f"images/{i}-clean.jpg")

            rand = random.randint(0, 2)

            if rand == 0:
                noised = add_noise("gaussian", image)
                attempted_denoise = standard_denoise(noised)

                save_image(noised, f"images/{i}-gaussian.jpg")
                save_image(attempted_denoise, f"images/{i}-gaussian-denoised.jpg")
            elif rand == 1:
                noised = add_noise("speckle", image)
                attempted_denoise = standard_denoise(noised)

                save_image(noised, f"images/{i}-speckle.jpg")
                save_image(attempted_denoise, f"images/{i}-speckle-denoised.jpg")
            else:
                noised = add_noise("salt & pepper", image)
                attempted_denoise = standard_denoise(noised)

                save_image(noised, f"images/{i}-salt-pepper.jpg")
                save_image(attempted_denoise, f"images/{i}-salt-pepper-denoised.jpg")
