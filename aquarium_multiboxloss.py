# Jaccard: 
# Hard negative mining: negative default box = 3times positve default bos
# Loss in regression task: MSE ->F.SmoothL1Loss
# Loss in classification (multi class): F.CrossEntropy
from aquarium_lib import *
from utils.box_utils import match
from collections import Counter


class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_threshold=0.5, neg_pos=3, device="cpu"):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_threshold = jaccard_threshold
        self.neg_pos = neg_pos
        self.device = device
    
    def forward(self, predictions, targets):
        #[num_batch_of_dataset_loader, 8732, num_box_info(xmax, ymax, xmin, ymin)=4], [8732, 4], [num_batch_of_dataset_loader, 8732, num_classes]
        loc_data, conf_data, dbox_list = predictions
        
        num_batch = loc_data.size(0) 
        num_dbox = loc_data.size(1) # 8732
        num_classes = conf_data.size(2) # 8( bg + classes)

        # [num_batch, num_dbox]
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        # [num_batch, num_dbox, 4]
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        for idx in range(num_batch):
            # [[xmin, ymin, xmax, ymax], ...]
            truths = targets[idx][:, :-1].to(self.device) #(xmin, ymin, xmax, ymax) BBox
            #[[label], ...]
            labels = targets[idx][:, -1].to(self.device) #label

            # [num_batch_of_dataset_loader, 8732, num_classes]
            dbox = dbox_list.to(self.device)
            variances = [0.1, 0.2]
            match(self.jaccard_threshold, truths, dbox, variances, labels, loc_t, conf_t_label, idx)

        #SmoothL1Loss
        # [num_batch, num_dbox]
        pos_mask = conf_t_label > 0
        # [num_batch, num_dbox, 4]
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # positive dbox, loc_data
        # [loc_data[pos_idx].size(0), 4]
        loc_p = loc_data[pos_idx].view(-1, 4)
        # [loc_data[pos_idx].size(0), 4]
        loc_t = loc_t[pos_idx].view(-1, 4)
        # []
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")

        #loss_conf
        #CrossEntropy
        # [num_batch*num_box, num_classes]
        batch_conf = conf_data.view(-1, num_classes)
        # [num_batch*num_box]
        loss_conf = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction="none")

        # hard negative mining
        # [num_batch, 1]
        num_pos = pos_mask.long().sum(1, keepdim=True)
        # [num_batch, 8732]
        loss_conf = loss_conf.view(num_batch, -1)

        _, loss_idx = loss_conf.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # idx_rank chính là thông số để biết được độ lớn loss nằm ở vị trí bao nhiêu

        num_neg = torch.clamp(num_pos*self.neg_pos, max=num_dbox)
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        #(num_batch, 8732) -> (num_batch, 8732, 21)
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)
        conf_t_pre = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1, num_classes)
        conf_t_label_ = conf_t_label[(pos_mask+neg_mask).gt(0)]
        loss_conf = F.cross_entropy(conf_t_pre, conf_t_label_, reduction="sum")

        # total loss = loss_loc + loss_conf
        N = num_pos.sum()
        # print(N)
        loss_loc = loss_loc/N
        loss_conf = loss_conf/N

        return loss_loc, loss_conf
        