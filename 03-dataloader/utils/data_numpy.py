import pathlib
import numpy as np
from PIL import Image


IMAGE_SIZE = 128
CHANNEL = 3


class MyDataLoader():

    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.data_list, self.labels = self._load_data_list()
        self.num_batches = int(np.ceil(len(self.data_list) / batch_size))

    def _load_data_list(self):
        data_list = []
        labels = []

        for d in pathlib.Path("data/catsanddogs").glob("*"):
            category = d.name
            labels.append(category)

            for f in d.glob("*.jpg"):
                data_list.append((str(f), labels.index(category)))

        return data_list, labels

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        indices = np.arange(len(self.data_list))
        np.random.shuffle(indices)

        for b in range(self.num_batches):
            start_index = b * self.batch_size
            end_index = min((b + 1) * self.batch_size, len(self.data_list))

            x_batch = np.zeros((end_index - start_index, IMAGE_SIZE, IMAGE_SIZE, CHANNEL))
            y_batch = np.zeros((end_index - start_index,))

            for i in range(start_index, end_index):
                path, label = self.data_list[indices[i]]
                image = self._load_data(path)
                image = self._preprocess(image)

                x_batch[i - start_index] = image
                y_batch[i - start_index] = label

            yield x_batch, y_batch

    def _load_data(self, path):
        image = Image.open(path)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        return image

    def _preprocess(self, image):
        # some preprocess, augmentation
        image = np.array(image)
        return image
