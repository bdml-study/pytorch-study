import numpy as np
import cv2
from settings import *


def preprocess(image, flip=False, distortion=False, noise=False):
    """
    image: 넘파이 배열이에요, PIL 이미지로 주지마세요!
    """
    
    image = (image.astype(np.float32) - 128)/256
    if len(image.shape) == 2:
        image = np.tile(np.expand_dims(image, axis=-1), [1, 1, 3])
    elif len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    if flip is True:
        image = _random_flip(image)
        
    if distortion is True:
        image = _color_distortion(image)
        
    if noise is True:
        image = _add_noise(image)
        
    return image


def _add_noise(image):
    if np.random.rand() < 0.5:
        image += 0.05*np.random.randn(*image.shape)
    return image


def _color_distortion(image):
    if np.random.rand() < 0.5:
        if np.random.rand() < 0.25:
            # red chanel distortion
            degree = np.random.rand() * 0.2
            endpoint = np.random.choice([-0.5, 0.5])
            image[..., 0] = image[..., 0]*(1 - degree) + endpoint*degree
        if np.random.rand() < 0.25:
            # green chanel distortion
            degree = np.random.rand() * 0.2
            endpoint = np.random.choice([-0.5, 0.5])
            image[..., 1] = image[..., 1]*(1 - degree) + endpoint*degree
        if np.random.rand() < 0.25:
            # blue chanel distortion
            degree = np.random.rand() * 0.2
            endpoint = np.random.choice([-0.5, 0.5])
            image[..., 2] = image[..., 2]*(1 - degree) + endpoint*degree
        
    return image


def _random_flip(image):
    if np.random.rand() < 0.5:
        image = image[:, ::-1].copy()
    return image