import numpy as np
from torchvision import transforms
from easyocr.imgproc import *
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import os

my_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class MyDataset(Dataset):

    def __init__(self, image_path, canvas_size=2560, transform=None):
        self.image_path = image_path
        self.canvas_size = canvas_size
        self.files = sorted(os.listdir(image_path))
        self.transform = my_transforms
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        img = loadImage(os.path.join(self.image_path, path))
        img_resize, target_ratio, size_heatmap = resize_aspect_ratio(img, self.canvas_size, interpolation=cv2.INTER_LINEAR)

        if self.transform:
            img_resize = self.transform(img_resize)
            
        return img_resize