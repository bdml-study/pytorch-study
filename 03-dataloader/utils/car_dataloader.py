import numpy as np
import os
import copy
from PIL import Image

DATA_PATH = "/mnt/d/data/car-object-detection/data/training_images/"
ANNOT_PATH = "/mnt/d/data/car-object-detection/data/train_solution_bounding_boxes.csv"
IMAGE_SIZE = 256
CHANNEL = 3


class CarDataLoader():

    def __init__(self, batch_size):
        self.data_list = self._load_data_list()

        self.batch_size = batch_size
        self.num_batches = int(np.ceil(len(self.data_list) / batch_size))

    def __len__(self):
        return self.num_batches

    def _load_data_list(self):
        data_list = []

        with open(ANNOT_PATH, "r") as f:
            lines = f.readlines()[1:]

        splited_lists = list(map(lambda line: line.strip().split(","), lines))
        
        filenames = list(map(lambda splited_list: splited_list[0], splited_lists))
        image_paths = list(map(lambda filename: os.path.join(DATA_PATH, filename), filenames))
        bboxes = list(map(lambda splited_list: list(map(float, splited_list[1:])), splited_lists))

        data_list.extend(list(zip(image_paths, bboxes)))
        return data_list

    def __iter__(self):
        data_list_copy = copy.deepcopy(self.data_list)
        np.random.shuffle(data_list_copy)

        for b in range(self.num_batches):
            start_index = b * self.batch_size
            end_index = min((b + 1) * self.batch_size, len(self.data_list))

            image_batch = np.zeros((end_index - start_index, CHANNEL, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            bbox_batch = np.zeros((end_index - start_index, 4), dtype=np.float32)

            for i in range(start_index, end_index):
                image_path, bbox = data_list_copy[i]
                image_batch[i - start_index], bbox_batch[i - start_index] = self._preprocess(image_path, bbox)

            yield image_batch, bbox_batch

    def _preprocess(self, path, bbox):
        image = Image.open(path)
        W, H = image.size

        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        bbox = [
            bbox[0] * IMAGE_SIZE/W,
            bbox[1] * IMAGE_SIZE/H,
            bbox[2] * IMAGE_SIZE/W,
            bbox[3] * IMAGE_SIZE/H
        ]

        image = self._image_preprocess(image)

        return image, bbox

    def _image_preprocess(self, image):
        image = np.array(image)
        image = (image.astype(np.float32) - 128) / 256
        image = np.transpose(image, [2, 0, 1])
        return image
