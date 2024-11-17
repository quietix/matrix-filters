from PIL import Image, ImageFilter, ImageEnhance
from config import logger
import numpy as np
from matplotlib.image import imread
from scipy import ndimage
from skimage import feature
import cv2


def apply_blur(original_img_path, filtered_img_path) -> bool:
    try:
        img = Image.open(original_img_path)
        img = img.filter(ImageFilter.BLUR)
        img.save(filtered_img_path)
        return True
    except Exception as e:
        logger.error(f"Error applying blur filter: {e}")
        return False


def apply_grey(original_img_path, filtered_img_path) -> bool:
    try:
        img = Image.open(original_img_path)
        img = img.convert("L")
        img.save(filtered_img_path)
        return True
    except Exception as e:
        logger.error(f"Error applying greyscale filter: {e}")
        return False


def apply_sharpen(original_img_path, filtered_img_path) -> bool:
    try:
        img = Image.open(original_img_path)
        sharpen_kernel = [
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0
        ]
        img = img.filter(ImageFilter.Kernel((3, 3), sharpen_kernel))
        img.save(filtered_img_path)
        return True
    except Exception as e:
        logger.error(f"Error applying sharpen filter: {e}")
        return False


def apply_emboss(original_img_path, filtered_img_path) -> bool:
    try:
        img = Image.open(original_img_path)
        emboss_kernel = [
            -2, -1, 0,
            -1, 1, 1,
            0, 1, 2
        ]
        img = img.filter(ImageFilter.Kernel((3, 3), emboss_kernel))
        img.save(filtered_img_path)
        return True
    except Exception as e:
        logger.error(f"Error applying emboss filter: {e}")
        return False


def apply_sobel(original_img_path, filtered_img_path) -> bool:
    try:
        original_image = imread(original_img_path)

        if len(original_image.shape) == 3:
            original_image = np.mean(original_image, axis=2)

        dx, dy = ndimage.sobel(original_image, axis=0), ndimage.sobel(original_image, axis=1)
        sobel_filtered_image = np.hypot(dx, dy)
        sobel_filtered_image = sobel_filtered_image / np.max(sobel_filtered_image)

        sobel_filtered_image = (sobel_filtered_image * 255).astype(np.uint8)
        Image.fromarray(sobel_filtered_image).save(filtered_img_path)

        return True

    except Exception as e:
        logger.error(f"Error applying Sobel filter: {e}")
        return False


def apply_canny(original_img_path, filtered_img_path) -> bool:
    try:
        original_image = imread(original_img_path)

        if len(original_image.shape) == 3:
            original_image = np.mean(original_image, axis=2)

        edges = feature.canny(original_image, sigma=1.0)
        Image.fromarray((edges * 255).astype(np.uint8)).save(filtered_img_path)

        return True

    except Exception as e:
        logger.error(f"Error applying Canny edge detection: {e}")
        return False


def apply_gaussian_blur(original_img_path, filtered_img_path) -> bool:
    try:
        img = Image.open(original_img_path)
        img = img.filter(ImageFilter.GaussianBlur(radius=5))
        img.save(filtered_img_path)
        return True
    except Exception as e:
        logger.error(f"Error applying Gaussian blur filter: {e}")
        return False


def apply_invert(original_img_path, filtered_img_path) -> bool:
    try:
        img = Image.open(original_img_path)
        img = Image.eval(img, lambda x: 255 - x)
        img.save(filtered_img_path)
        return True
    except Exception as e:
        logger.error(f"Error applying invert filter: {e}")
        return False


def apply_sepia(original_img_path, filtered_img_path) -> bool:
    try:
        img = Image.open(original_img_path)
        width, height = img.size
        pixels = img.load()

        for py in range(height):
            for px in range(width):
                r, g, b = img.getpixel((px, py))

                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)

                if tr > 255:
                    tr = 255
                if tg > 255:
                    tg = 255
                if tb > 255:
                    tb = 255

                pixels[px, py] = (tr, tg, tb)

        img.save(filtered_img_path)
        return True
    except Exception as e:
        logger.error(f"Error applying sepia filter: {e}")
        return False


def apply_vignette(original_img_path, filtered_img_path) -> bool:
    try:
        img = Image.open(original_img_path)
        width, height = img.size
        pixels = img.load()

        center_x, center_y = width / 2, height / 2
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

        for y in range(height):
            for x in range(width):
                dx = x - center_x
                dy = y - center_y
                distance = np.sqrt(dx**2 + dy**2)
                factor = 1 - (distance / max_distance)

                r, g, b = img.getpixel((x, y))
                r = int(r * factor)
                g = int(g * factor)
                b = int(b * factor)
                pixels[x, y] = (r, g, b)

        img.save(filtered_img_path)
        return True
    except Exception as e:
        logger.error(f"Error applying vignette effect: {e}")
        return False


def apply_posterize(original_img_path, filtered_img_path) -> bool:
    try:
        img = Image.open(original_img_path)
        img = img.convert("RGB")

        img = img.point(lambda p: p // 64 * 64)

        img.save(filtered_img_path)
        return True
    except Exception as e:
        logger.error(f"Error applying posterize effect: {e}")
        return False


def adjust_saturation(original_img_path, filtered_img_path, factor=1.5) -> bool:
    try:
        img = Image.open(original_img_path)
        img = img.convert("RGB")
        img = ImageEnhance.Color(img).enhance(factor)
        img.save(filtered_img_path)
        return True
    except Exception as e:
        logger.error(f"Error adjusting saturation: {e}")
        return False
