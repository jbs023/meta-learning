import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image, ImageOps

#TODO: Probably come up with a better dataset
class Omniglot(Dataset):
    def __init__(self, dataPath, transform=None, num_examples=None):
        np.random.seed(0)

        #Set up and download data here:
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)
        self.num_examples = num_examples if num_examples else sum([len(v) for k,v in self.datas.items()])

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        idx = 0

        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = list()
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1

        print("finish loading training dataset to memory")
        return datas, idx

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        label = None

        if index % 2 == 1:
            # Get image from same class
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            img1 = random.choice(self.datas[idx1])
            img2 = random.choice(self.datas[idx1])
        else:
            # Get image from different class
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            img1 = random.choice(self.datas[idx1])
            img2 = random.choice(self.datas[idx2])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))

class OmniglotMeta(Dataset):
    def __init__(self, dataPath, transform=None, way=5, shot=1):
        np.random.seed(0)

        #Set up and download data here:
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)
        self.way = way
        self.shot = shot        
        self.num_examples = sum([len(v) for k,v in self.datas.items()])

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        idx = 0

        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = list()
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1

        print("finish loading training dataset to memory")
        return datas, idx

    def __len__(self):
        return self.num_examples
        # return 1000

    def __getitem__(self, index):
        support_set = list()
        support_labels = list()
        query_set = list()
        query_labels = list()

        #First 5 are the support set
        for i in range(0, self.way):
            idx = random.randint(0, self.num_classes - 1)
            support_labels.append(i)
            for j in range(0, self.shot):
                img = random.choice(self.datas[idx])
                img_tensor = self.transform(img)
                support_set.append(img_tensor)

        #Last one is the query set
        support_class = random.randint(0, self.way - 1)
        idx = support_labels[support_class]
        query_labels.append(idx)

        for i in range(0, self.shot):
            img = random.choice(self.datas[idx])
            img_tensor = self.transform(img)
            query_set.append(img_tensor)

        data_dict = dict()
        data_dict["support"] = (torch.stack(support_set, 0), torch.LongTensor(support_labels))
        data_dict["query"] = (torch.stack(query_set), torch.LongTensor(query_labels))

        return data_dict
# test
if __name__=='__main__':
    omniglotTrain = Omniglot('./images_background')
    print(len(omniglotTrain))