# dataloader
# network -> SSD300
# loss -> MultiBoxLoss
# optimizer
# training, validation

from aquarium_dataset import AquariumDataset
from aquarium_lib import *
from aquarium_datapath import Datapath
from aquarium_transform import AquariumTransform
from aquarium_dataset import *
from aquarium_annotxt import *
from aquarium_model import SSD
from aquarium_multiboxloss import MultiBoxLoss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
torch.backends.cudnn.benchmark = True

# dataloader
root_path = r".\stuffs\aquarium_pretrain"
datapath = Datapath(root_path)
train_img_list, val_img_list, test_img_list = datapath.img_list()
train_anno_list, val_anno_list, test_anno_list = datapath.anno_list()
# train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(root_path)

classes = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']

color_mean = (104, 117, 123)
input_size = 300

#img_list, anno_list, phase, transform, anno_xml
train_dataset = AquariumDataset(train_img_list, train_anno_list, phase="train", transform=AquariumTransform(input_size, color_mean), anno_txt=AnnoTxt())
val_dataset = AquariumDataset(val_img_list, val_anno_list, phase="val", transform=AquariumTransform(input_size, color_mean), anno_txt=AnnoTxt())
test_dataset = AquariumDataset(test_img_list, test_anno_list, phase="test", transform=AquariumTransform(input_size, color_mean), anno_txt=AnnoTxt())

batch_size = 2
train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)
test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)
dataloader_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

# network
cfg = {
    "num_classes": 8, # 7 class + 1 background class
    "input_size": 300, #SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4], # Tỷ lệ khung hình cho source1->source6`
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264], # Size of default box
    "max_size": [60, 111, 162, 213, 264, 315], # Size of default box
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]]
}

net = SSD(phase="train", cfg=cfg)
vgg_weights = torch.load("./weights/vgg16_reducedfc.pth")
net.vgg.load_state_dict(vgg_weights)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# He init
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# MultiBoxLoss
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=device)

# optimizer
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# training, validation
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # move network to GPU
    net.to(device)

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    for epoch in range(num_epochs+1):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        print("---"*20)
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("---"*20)
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("(Training)")
            else:
                if (epoch+1) % 10 == 0:
                    net.eval() 
                    print("---"*10)
                    print("(Validation)")
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                # move to GPU
                # print("in")
                images = images.to(device)
                # print(images.shape)

                # targets = [ann.to(device) for ann in targets]
                # for t in targets:
                #         print(t[:,-1])
                # init optimizer
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase=="train"):
                    outputs = net(images)
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss.backward() # calculate gradient
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        optimizer.step() # update parameters

                        if (iteration % 10) == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print("Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec".format(iteration, loss.item(), duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
        t_epoch_end = time.time()
        print("---"*20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))           
        print("Duration: {:.4f} sec".format(t_epoch_end - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("./weights/ssd_logs.csv")
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), "./weights/ssd300_" + str(epoch+1) + ".pth")

num_epochs = 100
train_model(net, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)
