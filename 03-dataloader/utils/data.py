import pathlib
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = 128
CAT_DOG_DATA_PATH = "../data/catsanddogs"
CAR_DATA_PATH = "../data/car-object-detection/data/training_images/"
CAR_ANNOT_PATH = "../data/car-object-detection/data/train_solution_bounding_boxes.csv"


class CatsAndDogs(Dataset):

    def __init__(self):
        super(CatsAndDogs, self).__init__()

        self.data_list, self.labels = self._load_data_list()

    def _load_data_list(self):
        data_list = []
        labels = []

        for d in pathlib.Path(CAT_DOG_DATA_PATH).glob("*"):
            category = d.name
            labels.append(category)

            for f in d.glob("*.jpg"):
                data_list.append((str(f), labels.index(category)))

        return data_list, labels

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
        image = self._standardize(image)
        return image
        
    def _standardize(self, image):
        image = (image.astype(np.float32) - 128) / 256
        return image


class CarDataset(Dataset):

    def __init__(self):
        super(CarDataset, self).__init__()

        self.data_list = self._load_data_list()

    def _load_data_list(self):
        data_list = []

        with open(CAR_ANNOT_PATH, "r") as f:
            lines = f.readlines()[1:]

        splited_lists = list(map(lambda line: line.strip().split(","), lines))
        
        filenames = list(map(lambda splited_list: splited_list[0], splited_lists))
        image_paths = list(map(lambda filename: os.path.join(CAR_DATA_PATH, filename), filenames))
        bboxes = list(map(lambda splited_list: list(map(float, splited_list[1:])), splited_lists))

        data_list.extend(list(zip(image_paths, bboxes)))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path, bbox = self.data_list[index]
        image, bbox = self._load_data(path, bbox)
        return image, bbox

    def _load_data(self, path, bbox):
        image = Image.open(path)
        width, height = image.size
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image)

        bbox = [
            bbox[0] * (IMAGE_SIZE / width),
            bbox[1] * (IMAGE_SIZE / height),
            bbox[2] * (IMAGE_SIZE / width),
            bbox[3] * (IMAGE_SIZE / height),
        ]
        bbox = np.array(bbox, dtype=np.float32)

        image = self._preprocess(image)
        image = image.transpose(2, 0, 1)
        return image, bbox

    def _preprocess(self, image):
        image = self._standardize(image)
        return image
        
    def _standardize(self, image):
        image = (image.astype(np.float32) - 128) / 256
        return image
