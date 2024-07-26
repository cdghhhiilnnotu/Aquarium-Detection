from aquarium_annotxt import AnnoTxt
from aquarium_lib import *
from aquarium_datapath import *
from aquarium_transform import AquariumTransform

class AquariumDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_txt):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_txt = anno_txt

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_path = self.img_list[index]
        image = np.array(Image.open(image_path))

        anno_path = self.anno_list[index]
        anno = self.anno_txt(anno_path)

        # print(anno.shape)
        # print(anno_path)
        image, bboxes, labels = self.transform(image, self.phase, anno[:,1:], anno[:,0])

        # [width, height, channel] -> [channel, width, height]
        image = np.transpose(image, (2,1,0))
        image = torch.from_numpy(image)

        # [[xmin, ymin, xmax, ymax, label], ...]
        ground_truth = np.hstack((bboxes, np.expand_dims(labels, axis=1)))

        # (image, [[xmin, ymin, xmax, ymax, label], ...])
        return image, ground_truth

def my_collate_fn(batch):
    targets = []
    imgs = []

    # loop in batch of sample
    for sample in batch:
        imgs.append(torch.FloatTensor(sample[0])) # sample[0]=img
        targets.append(torch.FloatTensor(sample[1])) # sample[1]=annotation

    #[[3, 300, 300], [3, 300, 300], ...]
    # (num_of_batch, 3, 300, 300)
    imgs = torch.stack(imgs, dim=0)

    # ([(num_of_batch, 3, 300, 300)])
    return imgs, targets


if __name__ == "__main__":
    # print(Image.open(r'.\stuffs\aquarium_pretrain\train\images\IMG_2413_jpeg_jpg.rf.695815e23abdea80c043bb1cfd5a8a73.jpg').shape)
    root_path = r".\stuffs\aquarium_pretrain"
    datapath = Datapath(root_path)
    train_img_list, val_img_list, test_img_list = datapath.img_list()
    train_anno_list, val_anno_list, test_anno_list = datapath.anno_list()

    color_mean = (104, 117, 123)
    input_size = 300

    train_dataset = AquariumDataset(train_img_list, train_anno_list, phase="train",
                        transform=AquariumTransform(input_size, color_mean), anno_txt=AnnoTxt())
    
    val_dataset = AquariumDataset(val_img_list, val_anno_list, phase="val",
                        transform=AquariumTransform(input_size, color_mean), anno_txt=AnnoTxt())
    
    test_dataset = AquariumDataset(test_img_list, test_anno_list, phase="test",
                        transform=AquariumTransform(input_size, color_mean), anno_txt=AnnoTxt())

    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

    for _, target in train_dataloader:
        for t in target:
            if 7 in t[:,-1]:
                print(t[:,-1])

    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

    for _, target in val_dataloader:
        for t in target:
            if 7 in t[:,-1]:
                print(t[:,-1])

    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

    for _, target in test_dataloader:
        for t in target:
            if 7 in t[:,-1]:
                print(t[:,-1])







