from torch.utils.data import Dataset
import cv2
import pickle
import gzip
from scipy import ndimage
import os
import numpy as np
import random
import torch
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tqdm

equivalent_classes = {
    # Acevedo-20 dataset
    'BA': 'basophil',
    'EO': 'eosinophil',
    'ERB': 'erythroblast',
    'IG': "unknown",  # immature granulocytes
    'PMY': 'promyelocyte',  # immature granulocytes
    'MY': 'myelocyte',  # immature granulocytes
    'MMY': 'metamyelocyte',  # immature granulocytes
    'LY': 'lymphocyte_typical',
    'MO': 'monocyte',
    'NEUTROPHIL': "unknown",
    'BNE': 'neutrophil_banded',
    'SNE': 'neutrophil_segmented',
    'PLATELET': "unknown",
    
    # Matek-19 dataset
    'BAS': 'basophil',
    'EBO': 'erythroblast',
    'EOS': 'eosinophil',
    'KSC': 'smudge_cell',
    'LYA': 'lymphocyte_atypical',
    'LYT': 'lymphocyte_typical',
    'MMZ': 'metamyelocyte',
    'MOB': 'monocyte',  # monoblast
    'MON': 'monocyte',
    'MYB': 'myelocyte',
    'MYO': 'myeloblast',
    'NGB': 'neutrophil_banded',
    'NGS': 'neutrophil_segmented',
    'PMB': "unknown",
    'PMO': 'promyelocyte',
    
    # SYSU3H dataset
    'basophil' : 'basophil',
    'eosinophil' : 'eosinophil',
    'lymphocyte' : 'lymphocyte_typical',
    'monocyte' : 'monocyte',
    'neutrophil' :'neutrophil_segmented',
}

# Mapping unified class labels to numeric labels
label_map = {
    'basophil': 0,
    'eosinophil': 1,
    'erythroblast': 2,
    'myeloblast': 3,
    'promyelocyte': 4,
    'myelocyte': 5,
    'metamyelocyte': 6,
    'neutrophil_banded': 7,
    'neutrophil_segmented': 8,
    'monocyte': 9,
    'lymphocyte_typical': 10,
    'lymphocyte_atypical': 11,
    'smudge_cell': 12,
}

class Train_DataLoader(Dataset):
    def __init__(self, datasets_dir):
        self.datasets_names = ['Matek','Acevedo','SYSU3H']
        self.datasets_dir = datasets_dir
        self.mode = 'train'

        dataset_files = ['Matek-19.dat.gz','Acevedo-20.dat.gz','SYSU3H.dat.gz']
        samples = {}
        remove_key = []

        for file in dataset_files:
            print('loading ', file, '... ', end='')
            with gzip.open(os.path.join(self.datasets_dir, file), 'rb') as f:
                data = pickle.load(f)

                for d in data:
                    data[d]['dataset'] = file.split('-')[0]
                    
                    # Assign label based on the filename for non-SYSU3H datasets
                    if data[d]['dataset'] != 'SYSU3H':  
                        data[d]['label'] = d.split('_')[0]
                    
                    data[d]['masks'] = self.masks_processing(data[d]['masks'])
                
                samples = {**samples, **data}
            print('[done]')
        
        # Remove unknown classes and unnecessary samples
        keys = list(samples.keys())
        for s in keys:
            if equivalent_classes[samples[s]["label"]] == "unknown" or s in remove_key:
                samples.pop(s, None)

        self.samples = samples 
        self.keys = list(samples.keys())

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        key = self.keys[index]
        label = label_map[equivalent_classes[self.samples[key]['label']]]
        img = self.samples[key]['image'].copy()
        dataset = self.datasets_names.index(self.samples[key]['dataset'])
        masks = self.samples[key]['masks'].copy()
        feature = self.samples[key]['feature']
        
        # Preprocess the image data
        img = np.rollaxis(img, 2, 0)
        cropped_img = img.copy()
        cropped_img[~np.stack([masks]*3, axis=0).reshape(3, 224, 224)] = 128
        cropped_img = self.get_box_cropped(cropped_img, masks)
        cropped_img = zoom(cropped_img, (1, 128 / cropped_img.shape[1], 128 / cropped_img.shape[2]), order=3)
        cropped_img = torch.from_numpy(cropped_img.astype(np.float32)/255)
        feature = torch.from_numpy(feature.astype(np.float32))
        
        return feature, cropped_img, label, dataset
    
    # Process masks to keep only the largest connected component
    def masks_processing(self, masks):
        labeled_arr, num_features = ndimage.label(masks)
        sizes = ndimage.sum(masks, labeled_arr, range(num_features+1))
        max_label = np.argmax(sizes[1:]) + 1
        output = np.zeros_like(masks, dtype=bool)
        output[labeled_arr == max_label] = True
        return output
    
    # Crop the image based on the bounding box of the mask
    def get_box_cropped(self, image, masks):
        img = image.copy()
        h, w = masks.shape
        y1, y2, x1, x2 = -1, -1, -1, -1

        for i in range(h):
            if True in masks[i]:
                y1 = i
                break
        for i in range(h-1, -1, -1):
            if True in masks[i]:
                y2 = i
                break
        for i in range(w):
            if True in masks[:, i]:
                x1 = i
                break
        for i in range(w-1, -1, -1):
            if True in masks[:, i]:
                x2 = i
                break
        
        # Crop the image and add padding
        img = img[:, y1:y2, x1:x2]
        f = np.full((3, 10, img.shape[2]), 128, dtype=np.uint8)
        img = np.concatenate([f, img, f], axis=1)
        f = np.full((3, img.shape[1], 10), 128, dtype=np.uint8)
        img = np.concatenate([f, img, f], axis=2)
        
        return img  
    
if __name__=='__main__':
    dataset_dir = './SAM'
    train_data_loader = Train_DataLoader(dataset_dir)

    data_loader = DataLoader(train_data_loader, batch_size=64, shuffle=True)