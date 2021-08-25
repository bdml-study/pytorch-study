import pathlib
import numpy as np
import cv2
from PIL import Image
from settings import *
from torch.utils.data import Dataset


class CatsAndDogs(Dataset):

    def __init__(self, mode="train", noise=False, distortion=False, crop=False):
        super(CatsAndDogs, self).__init__()

        if mode != "train" and mode != "test":
            raise ValueError("INVALID mode: " + mode)

        self.mode = mode
        self.noise = noise
        self.distortion = distortion
        self.crop = crop
        self.data_list, self.labels = self._load_data_list(mode)

    def _load_data_list(self, mode):
        data_list = []
        labels = []

        for d in pathlib.Path(DATA_PATH).glob("*"):
            if d.is_dir():
                labels.append(d.name)
                data_list.append([])

                for f in d.glob("*.jpg"):
                    data_list[-1].append((str(f), float(labels.index(d.name))))

        flattened_data_list = []

        for i in range(len(data_list)):
            data_length = len(data_list[i])
            if mode == "train":
                data_list[i] = data_list[i][:int(data_length*0.7)]
            elif mode == "test":
                data_list[i] = data_list[i][int(data_length*0.7):]

            flattened_data_list.extend(data_list[i])

        return flattened_data_list, labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path, label = self.data_list[index]
        image = self._load_image(path)
        return image, label

    def _load_image(self, path):
        image = Image.open(path)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image)
        image = self._preprocess(image)
        image = image.transpose(2, 0, 1)
        return image

    def _preprocess(self, image):
        image = (image.astype(np.float32) - 128)/256
        if len(image.shape) == 2:
            image = np.tile(np.expand_dims(image, axis=-1), [1, 1, 3])
        elif len(image.shape) == 3 and image.shape[-1] == 4:
            image = image[..., :3]
            
        if self.crop is True:
            image = self._random_crop(image)
            
        if self.distortion is True:
            image = self._color_distortion(image)
            
        if self.noise is True:
            image = self._add_noise(image)
            
        return image
    
    def _add_noise(self, image):
        if np.random.rand() < 0.5:
            image += 0.05*np.random.randn(*image.shape)
        return image
    
    def _color_distortion(self, image):
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
    
    def _random_crop(self, image):
        x, y = np.random.randint(0, IMAGE_SIZE//4, size=2)
        w, h = np.random.randint(3*(IMAGE_SIZE//4), IMAGE_SIZE, size=2)
        
        image = image[y:y+h, x:x+w]
        image = cv2.resize(image, dsize=(IMAGE_SIZE, IMAGE_SIZE))
        return image

