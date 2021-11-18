import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image, ImageOps

class OmniglotTrain(Dataset):
    def __init__(self, dataPath, transform=None, distortions=False, num_examples=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)
        self.transform = transform
        self.distortions = distortions
        self.datas, self.num_classes = self.loadToMem(dataPath)
        self.num_examples = num_examples if num_examples else sum([len(v) for k,v in self.datas.items()])

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        idx = 0

        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    if self.distortions:
                        #Apply 8 affine distortions to each image
                        original_image = Image.open(filePath).convert('L')
                        for _ in range(0, 8):
                            distortedImg = self.affine_distortions(original_image)
                            datas[idx].append(distortedImg.convert('L'))
                    else:
                        datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1

        print("finish loading training dataset to memory")
        return datas, idx

    def affine_distortions(self, img):
        width, height = img.size

        #Sheer Horizontal
        if random.uniform(0, 1) > 0.5:
            rho_x = random.uniform(-0.3, 0.3)
            # rho_x = math.radians(rho_x)
            img = ImageOps.invert(img.convert('RGB'))
            img = img.transform(img.size, Image.AFFINE, (1, rho_x, 0, 0, 1, 0))
            img = ImageOps.invert(img.convert('RGB'))

        #Sheer Vertical
        if random.uniform(0, 1) > 0.5:
            rho_y = random.uniform(-0.3, 0.3)
            # rho_y - math.radians(rho_y)
            img = ImageOps.invert(img.convert('RGB'))
            img = img.transform(img.size, Image.AFFINE, (1, 0, 0, rho_y, 1, 0))
            img = ImageOps.invert(img.convert('RGB'))

        #Scale X
        if random.uniform(0, 1) > 0.5:
            s_x = random.uniform(0.8, 1.2)
        else:
            s_x = 1

        #Scale Y
        if random.uniform(0, 1) > 0.5:
            s_y = random.uniform(0.8, 1.2)
        else:
            s_y = 1
        # img = img.resize((round(s_x * img.size[0]), round(s_y * img.size[1])), Image.ANTIALIAS)
        img = ImageOps.invert(img.convert('RGB'))
        img = img.resize((round(s_x * img.size[0]), round(s_y * img.size[1])))
        img = ImageOps.invert(img.convert('RGB'))

        #Rotate Image
        if random.uniform(0, 1) > 0.5:
            theta = random.uniform(-10.0, 10.0)
            img = img.rotate(theta, expand=1, fillcolor=(255, 255, 255))

        #Translate X
        if random.uniform(0, 1) > 0.5:
            t_x = random.uniform(-2, 2)
            img = ImageOps.invert(img.convert('RGB'))
            img = img.transform(img.size, Image.AFFINE, (1, 0, t_x, 0, 1, 0))
            img = ImageOps.invert(img.convert('RGB'))

        #Translate Y
        if random.uniform(0, 1) > 0.5:
            t_y = random.uniform(-2, 2)
            img = ImageOps.invert(img.convert('RGB'))
            img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, t_y))
            img = ImageOps.invert(img.convert('RGB'))

        #Crop to final size
        img = ImageOps.invert(img.convert('RGB'))
        img = img.crop((0, 0, width, height))
        img = ImageOps.invert(img.convert('RGB'))

        return img

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        label = None

        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class OmniglotTest(Dataset):
    def __init__(self, dataPath, transform=None, times=200, way=20):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2


# test
if __name__=='__main__':
    omniglotTrain = OmniglotTrain('./images_background')
    print(len(omniglotTrain))