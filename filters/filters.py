from config import logger
import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import convolve
from PIL import Image, ImageOps
import cv2
from typing import Optional
from animegan import AnimeGAN


hayao_64 = "hayao_64"
hayao_60 = "hayao_60"
paprika_54 = "paprika_54"
shinkai_53 = "shinkai_53"

model_paths = {hayao_64: r"models\Hayao_64.onnx",
               hayao_60: r"models\Hayao_60.onnx",
               paprika_54: r"models\Paprika_54.onnx",
               shinkai_53: r"models\Shinkai_53.onnx"}


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


def _prepare_img_for_processing(img_path) -> np.ndarray:
    with Image.open(img_path) as img:
        img = ImageOps.exif_transpose(img)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = np.array(img, dtype=np.float32) / 255
        return img


def _get_channels(img: np.ndarray) -> Optional[int]:
    return img.shape[2] if img.ndim == 3 else None


@_error_decorator("Error in applying gray filter")
def turn_gray(original_img_path, filtered_img_path):
    img_plt = _prepare_img_for_processing(original_img_path)
    gray = np.mean(img_plt, axis=2)
    plt.imsave(filtered_img_path, gray,  cmap='gray')


@_error_decorator("Error in inverting colors")
def invert_colors(original_img_path, filtered_img_path):
    new_img_plt = _prepare_img_for_processing(original_img_path)

    if new_img_plt.dtype != np.uint8:
        new_img_plt = (new_img_plt * 255).astype(np.uint8)

    new_img_plt = new_img_plt[:, :, :3]
    new_img_plt = 255 - new_img_plt

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in leaving only red")
def leave_only_red(original_img_path, filtered_img_path):
    new_img_plt = _prepare_img_for_processing(original_img_path)

    new_img_plt[:, :, 1], new_img_plt[:, :, 2] = 0, 0

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in leaving only green")
def leave_only_green(original_img_path, filtered_img_path):
    new_img_plt = _prepare_img_for_processing(original_img_path)

    new_img_plt[:, :, 0], new_img_plt[:, :, 2] = 0, 0

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in leaving only blue")
def leave_only_blue(original_img_path, filtered_img_path):
    new_img_plt = _prepare_img_for_processing(original_img_path)

    new_img_plt[:, :, 0], new_img_plt[:, :, 1] = 0, 0

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in applying box blur")
def apply_box_blur(original_img_path, filtered_img_path):
    new_img_plt = _prepare_img_for_processing(original_img_path)

    blur_kernel = np.ones((3, 3)) / 9

    for channel in range(3):
        new_img_plt[:, :, channel] = convolve(new_img_plt[:, :, channel], blur_kernel)

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in applying Gaussian blur")
def apply_blur_gaussian(original_img_path, filtered_img_path):
    new_img_plt = _prepare_img_for_processing(original_img_path)

    blur_kernel = np.array(((1, 4, 6, 4, 1),
                            (4, 16, 24, 16, 4),
                            (6, 24, 36, 24, 6),
                            (4, 16, 24, 16, 4),
                            (1, 4, 6, 4, 1),)) / 256

    for channel in range(3):
        new_img_plt[:, :, channel] = convolve(new_img_plt[:, :, channel], blur_kernel)

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in applying sharpen")
def apply_sharpen(original_img_path, filtered_img_path):
    new_img_plt = _prepare_img_for_processing(original_img_path)

    blur_kernel = np.array(((0, -1, 0),
                            (-1, 5, -1),
                            (0, -1, 0)), dtype=np.float32)

    channels = _get_channels(new_img_plt)

    if channels:
        for channel in range(channels):
            new_img_plt[:, :, channel] = convolve(new_img_plt[:, :, channel], blur_kernel)
    else:
        new_img_plt[:, :] = convolve(new_img_plt[:, :], blur_kernel)

    new_img_plt = np.clip(new_img_plt, 0, 1)

    plt.imsave(filtered_img_path, new_img_plt)


@_error_decorator("Error in applying sobel")
def apply_sobel(original_img_path, filtered_img_path):
    img_plt = _prepare_img_for_processing(original_img_path)
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
    img_plt = _prepare_img_for_processing(original_img_path)
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


@_error_decorator("Error in applying sharpen")
def cartoonize(original_img_path, filtered_img_path):
    image = cv2.imread(original_img_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=9)

    color = cv2.bilateralFilter(image, d=9, sigmaColor=200, sigmaSpace=200)

    cartoon = cv2.bitwise_and(color, color, mask=edges)

    cv2.imwrite(filtered_img_path, cartoon)


@_error_decorator("Error in applying Hayao-64")
def apply_animization_hayao_64(original_img_path, filtered_img_path):
    model_path = model_paths.get(hayao_64)

    if model_path is None:
        raise FileNotFoundError(f"Model path for {hayao_64} not found.")

    animegan = AnimeGAN(model_path=model_path, downsize_ratio=1)
    img = cv2.imread(original_img_path)
    img = animegan(img)
    cv2.imwrite(filtered_img_path, img)


@_error_decorator("Error in applying Hayao-60")
def apply_animization_hayao_60(original_img_path, filtered_img_path):
    model_path = model_paths.get(hayao_60)

    if model_path is None:
        raise FileNotFoundError(f"Model path for {hayao_60} not found.")

    animegan = AnimeGAN(model_path=model_path, downsize_ratio=1)
    img = cv2.imread(original_img_path)
    img = animegan(img)
    cv2.imwrite(filtered_img_path, img)


@_error_decorator("Error in applying Paprika-54")
def apply_animization_paprika_54(original_img_path, filtered_img_path):
    model_path = model_paths.get(paprika_54)

    if model_path is None:
        raise FileNotFoundError(f"Model path for {paprika_54} not found.")

    animegan = AnimeGAN(model_path=model_path, downsize_ratio=1)
    img = cv2.imread(original_img_path)
    img = animegan(img)
    cv2.imwrite(filtered_img_path, img)


@_error_decorator("Error in applying Shinkai-53")
def apply_animization_shinkai_53(original_img_path, filtered_img_path):
    model_path = model_paths.get(shinkai_53)

    if model_path is None:
        raise FileNotFoundError(f"Model path for {shinkai_53} not found.")

    animegan = AnimeGAN(model_path=model_path, downsize_ratio=1)
    img = cv2.imread(original_img_path)
    img = animegan(img)
    cv2.imwrite(filtered_img_path, img)