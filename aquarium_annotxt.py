from aquarium_lib import *
from aquarium_datapath import *

class AnnoTxt:
    def __init__(self):
        self.classes = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']

    def __call__(self, txt_path):
        ret = []
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                stripped_line = line.strip()
                if stripped_line:
                    anno_infor = stripped_line.split(' ')
                    anno_infor = np.array(anno_infor, dtype=np.float32)
                    anno = []
                    anno.append(anno_infor[0])
                    anno.append(anno_infor[1] - anno_infor[3]/2)
                    anno.append(anno_infor[2] - anno_infor[4]/2)
                    anno.append(anno_infor[1] + anno_infor[3]/2)
                    anno.append(anno_infor[2] + anno_infor[4]/2)
                    ret.append(list(anno))
        
        # [[label, xmin, ymin, xmax, ymax], [], ...]
        return np.array(ret, dtype=np.float32)


if __name__ == "__main__":
    anno = AnnoTxt()

    datapath = Datapath(r".\stuffs\aquarium_pretrain")
    idx = random.randint(0, len(datapath.anno_list()[0])-1)

    print(datapath.anno_list()[0][idx])
    print(datapath.img_list()[0][idx])
    print(anno(datapath.anno_list()[0][idx]))







