from aquarium_lib import *
from aquarium_datapath import Datapath


datapath = Datapath(r".\stuffs\aquarium_pretrain")
anno_list = datapath.anno_list()
img_list = datapath.img_list()

label = []
for anno_phase in anno_list:
    for anno_file in anno_phase:
        with open(anno_file, 'r') as file:
            print(anno_file)
            lines = file.readlines()
            for line in lines:
                stripped_line = line.strip()
                if stripped_line:
                    anno_infor = stripped_line.split(' ')
                    anno_infor = np.array(anno_infor, dtype=np.float32)
                    # label.append(anno_infor[0])
                    print(anno_infor)

label = np.array(label)
print(np.unique(label))



