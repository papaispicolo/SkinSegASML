from torch.utils.data import Dataset
from torchvision import transforms
import glob
import matplotlib.pyplot as plt
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, imageDir, maskDir, transforms=None, cache=False):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.cache = cache
        self.imagePaths = sorted(glob.glob(imageDir + "/*"))
        # print(self.imagePaths)
        self.maskPaths = sorted(glob.glob(maskDir + "/*"))
        # print(self.imagePaths)
        self.transforms = transforms
        if self.cache:
            self.cache_storage = [None] * self.__len__()

    def __len__(self):
        # return the number of total samples contained in t he dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        if self.cache_storage[idx] is None:
            imagePath = self.imagePaths[idx]
            #print(imagePath)
            # load the image from disk, swap its channels from BGR to RGB,
            # and read the associated mask from disk in grayscale mode
            image = cv2.imread(imagePath)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.maskPaths[idx], 0)
            # check to see if we are applying any transformations
            if self.transforms is not None:
                # apply the transformations to both image and its mask
                image = self.transforms[0](image)
                mask = self.transforms[1](mask)
                # return a tuple of the image and its mask
                self.cache_storage[idx] = (image, mask)
                return (image, mask)
        else:
            return self.cache_storage[idx]
        