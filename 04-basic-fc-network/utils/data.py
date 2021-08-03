import pathlib
import numpy as np
from PIL import Image
from settings import *
from torch.utils.data import Dataset


class CatsAndDogs(Dataset):

    def __init__(self, mode="train"):
        super(CatsAndDogs, self).__init__()

        if mode != "train" and mode != "test":
            raise ValueError("INVALID mode: " + mode)

        self.mode = mode
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
        image = (image.astype(np.float32) - 128)/256
        image = image.reshape(-1)
        return image
