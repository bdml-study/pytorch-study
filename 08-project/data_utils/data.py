import pathlib
import numpy as np
from PIL import Image
from settings import *
from preprocess import preprocess
from torch.utils.data import Dataset


class FaceMaskDetectorData(Dataset):

    def __init__(self, mode="train", noise=False, distortion=False, crop=False, flip=False):
        super(FaceMaskDetectorData, self).__init__()

        if mode != "train" and mode != "test" and mode != "valid":
            raise ValueError("INVALID mode: " + mode)

        self.mode = mode
        self.noise = noise
        self.distortion = distortion
        self.flip = flip
        self.crop = crop
        self.labels = ["WithoutMask", "WithMask"]
        self.data_list = self._load_data_list(mode)

    def _load_data_list(self, mode):
        data_list = []

        if mode == "train":
            mode = "Train"
        elif mode == "valid":
            mode = "Validation"
        elif mode == "test":
            mode = "Test"

        for d in pathlib.Path(f"{DATA_PATH}/{mode}").glob("*"):
            if d.is_dir() and d.name in self.labels:
                for f in d.glob("*.png"):
                    data_list.append((str(f), float(self.labels.index(d.name))))

        return data_list

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
        image = preprocess(image)
        image = image.transpose(2, 0, 1)
        return image


