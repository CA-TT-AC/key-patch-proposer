import json
from PIL import Image
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from util.datasets import build_transform

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: The root directory where the image is stored, e.g. '/path/to/imagenet/train/'
        transform: transform operation applied to each image
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = [] 
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png') or img_name.endswith('.JPEG'):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        img_name = os.path.basename(img_path).split('.')[0]
        
        return image, img_name
    
class ImageNetDatasetPerClass(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: The root directory where the image is stored, e.g. '/path/to/imagenet/train/nxxxxxx'
        transform: transform operation applied to each image
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = [] 
        class_dir = self.root_dir
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.png') or img_name.endswith('.JPEG'):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, self.root_dir.split('/')[-1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 提取图片名称 (不含扩展名)
        img_name = os.path.basename(img_path).split('.')[0]
        
        return image, img_name

class ImagenetteKPP(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: The root directory where the image is stored, e.g. '/path/to/imagenette/train/'
        transform: transform operation applied to each image
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []  # 存储 (图片路径, 类别索引, patch_ids) 元组
        # 构建样本列表
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            json_path = os.path.join(class_dir, "patch_ids.json")

            # 读取json为dict
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_dict = json.load(f)

                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png') or img_name.endswith('.JPEG'):
                        img_path = os.path.join(class_dir, img_name)
                        image_base = os.path.splitext(img_name)[0]  # 移除扩展名

                        # 从JSON中索引patch_ids
                        assert image_base in json_dict
                        patch_id = json_dict[image_base]["patch"]
                        self.samples.append((img_path, class_idx, patch_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx, patch_ids = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        patch_ids = torch.tensor(patch_ids)
        return image, class_idx, patch_ids
    
def build_dataset_KPP(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = ImagenetteKPP(root, transform=transform)
    print(dataset)
    return dataset
