from config import logger
import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import convolve
from PIL import Image, ImageOps
import cv2


def _error_decorator(error_message: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
                return True
            except Exception as e:
                logger.error(f"{error_message}: {e}")
                return False
        return wrapper
    return decorator


def _get_img_stripped_from_exif(img_path) -> np.ndarray:
    with Image.open(img_path) as img:
        img = ImageOps.exif_transpose(img)
        return np.array(img)


@_error_decorator("Error in applying gray filter")
def turn_gray(original_img_path, filtered_img_path):
    img_plt = _get_img_stripped_from_exif(original_img_path)
    gray = np.mean(img_plt, axis=2)
    plt.imsave(filtered_img_path, gray,  cmap='gray')


@_error_decorator("Error in inverting colors")
def invert_colors(original_img_path, filtered_img_path):
    new_img_plt = _get_img_stripped_from_exif(original_img_path)

    if new_img_plt.dtype != np.uint8:
        new_img_plt = (new_img_plt * 255).astype(np.uint8)

    new_img_plt = new_img_plt[:, :, :3]
    new_img_plt = 255 - new_img_plt

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in leaving only red")
def leave_only_red(original_img_path, filtered_img_path):
    new_img_plt = _get_img_stripped_from_exif(original_img_path)

    new_img_plt[:, :, 1], new_img_plt[:, :, 2] = 0, 0

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in leaving only green")
def leave_only_green(original_img_path, filtered_img_path):
    new_img_plt = _get_img_stripped_from_exif(original_img_path)

    new_img_plt[:, :, 0], new_img_plt[:, :, 2] = 0, 0

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in leaving only blue")
def leave_only_blue(original_img_path, filtered_img_path):
    new_img_plt = _get_img_stripped_from_exif(original_img_path)

    new_img_plt[:, :, 0], new_img_plt[:, :, 1] = 0, 0

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in applying blur")
def apply_blur(original_img_path, filtered_img_path):
    new_img_plt = _get_img_stripped_from_exif(original_img_path)

    blur_kernel = np.ones((3, 3)) / 9

    for channel in range(3):
        new_img_plt[:, :, channel] = convolve(new_img_plt[:, :, channel], blur_kernel)

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in applying blur")
def apply_blur(original_img_path, filtered_img_path):
    new_img_plt = _get_img_stripped_from_exif(original_img_path)

    blur_kernel = np.array(((1, 4, 6, 4, 1),
                            (4, 16, 24, 16, 4),
                            (6, 24, 36, 24, 6),
                            (4, 16, 24, 16, 4),
                            (1, 4, 6, 4, 1),)) / 256

    for channel in range(3):
        new_img_plt[:, :, channel] = convolve(new_img_plt[:, :, channel], blur_kernel)

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in applying sobel")
def apply_sobel(original_img_path, filtered_img_path):
    img_plt = _get_img_stripped_from_exif(original_img_path)
    grayscale_img = np.mean(img_plt, axis=2)

    gx_kernel = np.array(((1, 0, -1),
                          (2, 0, -2),
                          (1, 0, -1)))
    gy_kernel = np.array(((1, 2, 1),
                          (0, 0, 0),
                          (-1, -2, -1)))

    gx = convolve(grayscale_img, gx_kernel)
    gy = convolve(grayscale_img, gy_kernel)
    sobel_filtered_image = np.hypot(gx, gy)

    plt.imsave(filtered_img_path, sobel_filtered_image, cmap='gray')


@_error_decorator("Error in applying colorized sobel")
def apply_colorized_sobel(original_img_path, filtered_img_path):
    img_plt = _get_img_stripped_from_exif(original_img_path)
    grayscale_img = np.mean(img_plt, axis=2)

    gx_kernel = np.array(((1, 0, -1),
                          (2, 0, -2),
                          (1, 0, -1)))
    gy_kernel = np.array(((1, 2, 1),
                          (0, 0, 0),
                          (-1, -2, -1)))

    gx = convolve(grayscale_img, gx_kernel)
    gy = convolve(grayscale_img, gy_kernel)

    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx) * 180 / np.pi

    magnitude_normalized = magnitude / np.max(magnitude)
    angle_normalized = (angle + 180) / 360.0

    hsv_image: np.ndarray = np.zeros_like(img_plt, dtype=np.float32)

    hsv_image[..., 0] = angle_normalized
    hsv_image[..., 1] = 1.0
    hsv_image[..., 2] = magnitude_normalized

    hsv_image_uint8 = (hsv_image * 255).astype(np.uint8)
    rgb_image = cv2.cvtColor(hsv_image_uint8, cv2.COLOR_HSV2RGB)

    plt.imsave(filtered_img_path, rgb_image)
