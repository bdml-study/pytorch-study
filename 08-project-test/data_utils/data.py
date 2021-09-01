from settings import *
from preprocess import preprocess
from torch.utils.data import Dataset


class FaceMaskDetectorData(Dataset):

    def __init__(self, mode="train", noise=False, distortion=False, crop=False, flip=False):
        super(FaceMaskDetectorData, self).__init__()

        if mode != "train" and mode != "test" and mode != "valid":
            raise ValueError(f"INVALID mode: '{mode}'")

        self.mode = mode
        self.noise = noise
        self.distortion = distortion
        self.flip = flip
        self.crop = crop


