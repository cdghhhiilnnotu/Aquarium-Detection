from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
    PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, \
    ToPercentCoords, Resize, SubtractMeans

from aquarium_lib import *

class AquariumTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(), # convert image from int to float 32
                ToAbsoluteCoords(), # back annotation to normal type
                PhotometricDistort(), # change color by random
                Expand(color_mean), 
                RandomSampleCrop(), # randomcrop image
                RandomMirror(), # mirror image
                ToPercentCoords(), # standize annotation data to [0-1]
                Resize(input_size),
                SubtractMeans(color_mean) # Subtract mean of BGR
            ]), 
            "val": Compose([
                ConvertFromInts(), # convert image from int to float 32
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
            , 
            "test": Compose([
                ConvertFromInts(), # convert image from int to float 32
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        # (img after transformed, boxes transformed with image, label of box)
        return self.data_transform[phase](img, boxes, labels)
