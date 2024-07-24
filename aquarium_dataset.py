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

        image, bboxes, labels = self.transform(image, self.phase, anno[:,1:], anno[:,0])

        # [width, height, channel] -> [channel, width, height]
        image = torch.from_numpy(image[:,:,(2,1,0)])

        # [[label, xmin, ymin, xmax, ymax], ...]
        ground_truth = np.hstack((bboxes, np.expand_dims(labels, axis=1)))

        # (image, [[label, xmin, ymin, xmax, ymax], ...])
        return image, ground_truth
    
def my_collate_fn(batch):
    targets = []
    imgs = []

    # loop in batch of sample
    for sample in batch:
        imgs.append(torch.FloatTensor(sample[0])) # sample[0]=img
        print(sample[1])
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
    
    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

    batch_iter = iter(train_dataloader)
    images, targets = next(batch_iter)
    print(images.size()) 
    print(len(targets))
    print(targets[0].size())






