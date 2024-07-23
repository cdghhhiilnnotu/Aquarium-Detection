from aquarium_lib import *

class AquariumDataset():
    def __init__(self, img_list, anno_list, phase, transform, anno_txt):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_txt = anno_txt

    



