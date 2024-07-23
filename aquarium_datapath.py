from aquarium_lib import *

class Datapath:
    def __init__(self, root_path):
        self.root_path = root_path

        # '{root_path}\train\{(labels or images)\}'
        self.anno_train_dir = osp.join(root_path, "train", "labels")
        self.img_train_dir = osp.join(root_path, "train", "images")

        # '{root_path}\val\{(labels or images)\}'
        self.anno_val_dir = osp.join(root_path, "valid", "labels")
        self.img_val_dir = osp.join(root_path, "valid", "images")

        # '{root_path}\test\{(labels or images)\}'
        self.anno_test_dir = osp.join(root_path, "test", "labels")
        self.img_test_dir = osp.join(root_path, "test", "images")

    def anno_list(self):
        self.anno_train_list = [osp.join(self.anno_train_dir, file_name) for file_name in os.listdir(self.anno_train_dir)]
        self.anno_val_list = [osp.join(self.anno_val_dir, file_name) for file_name in os.listdir(self.anno_val_dir)]
        self.anno_test_list = [osp.join(self.anno_test_dir, file_name) for file_name in os.listdir(self.anno_test_dir)]

        # ([anno_train_list],[anno_val_list],[anno_test_list])
        return self.anno_train_list, self.anno_val_list, self.anno_test_list
    
    def img_list(self):
        self.img_train_list = [osp.join(self.img_train_dir, file_name) for file_name in os.listdir(self.img_train_dir)]
        self.img_val_list = [osp.join(self.img_val_dir, file_name) for file_name in os.listdir(self.img_val_dir)]
        self.img_test_list = [osp.join(self.img_test_dir, file_name) for file_name in os.listdir(self.img_test_dir)]

        # ([img_train_list],[img_val_list],[img_test_list])
        return self.img_train_list, self.img_val_list, self.img_test_list
    


if __name__ == "__main__":
    datapath = Datapath(r".\stuffs\aquarium_pretrain")
    print(len(datapath.img_list()[0]))






