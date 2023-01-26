import os
import random
import shutil

from meta_learn.dataset import Omniglot, OmniglotMeta
from google_drive_downloader import GoogleDriveDownloader as gdd
from torchvision.transforms import Compose, ToTensor

def download_omniglot(data_path):
    '''Setup Omniglot meta-dataset'''
    # Download Omniglot Dataset
    omniglot_path = f"{data_path}/omniglot_resized.zip"
    if not os.path.isdir(omniglot_path):
        gdd.download_file_from_google_drive(file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                            dest_path=omniglot_path,
                                            unzip=True)

    character_folders = [os.path.join(data_path, family, character)
                            for family in os.listdir(data_path)
                            if os.path.isdir(os.path.join(data_path, family))
                            for character in os.listdir(os.path.join(data_path, family))
                            if os.path.isdir(os.path.join(data_path, family, character))]

    random.Random(0).shuffle(character_folders) #Deterministic random shuffle
    num_train = int(.8*len(character_folders))
    metatrain_character_folders = character_folders[: num_train]
    metatest_character_folders = character_folders[num_train :]

    #Divide data into train and test
    if not (os.path.exists(f"{data_path}/omni_train")):
        os.makedirs(f"{data_path}/omni_train")
        os.makedirs(f"{data_path}/omni_test")

        for folder in metatrain_character_folders:
            folder_name = folder.split("/")[-1]
            if not os.path.exists(f"{data_path}/omni_train/{folder_name}"):
                shutil.copytree(folder, f"{data_path}/omni_train/{folder_name}")

        for folder in metatest_character_folders:
            folder_name = folder.split("/")[-1]
            if not os.path.exists(f"{data_path}/omni_test/{folder_name}"):
                shutil.copytree(folder, f"{data_path}/omni_test/{folder_name}")
 
    return f"{data_path}/omni_train", f"{data_path}/omni_test"

def get_siamese_omniglot(data_path):
    '''Setup Omniglot meta-dataset'''
    # Download Omniglot Dataset
    train_path, test_path = download_omniglot(data_path)
 
    transform = Compose([ToTensor()])    
    train_set = Omniglot(train_path, transform)
    test_set = Omniglot(test_path, transform)

    return train_set, test_set

def get_meta_omniglot(data_path, way, shot):
    '''Setup Omniglot meta-dataset'''
    # Download Omniglot Dataset
    train_path, test_path = download_omniglot(data_path)
 
    transform = Compose([ToTensor()])    
    train_set = OmniglotMeta(train_path, transform, way, shot)
    test_set = OmniglotMeta(test_path, transform, way, shot)

    return train_set, test_set