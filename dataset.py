import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

class HelmetDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.img_labels = self.load_annotations()
        self.image_paths = self.load_image_paths()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Đọc ảnh dưới dạng PIL Image
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_annotations(self):
        annotation_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.csv')]

        annotations = []
        for file in annotation_files:
            file_path = os.path.join(self.annotations_dir, file)
            df = pd.read_csv(file_path)
            annotations.append(df)

        annotations = pd.concat(annotations, ignore_index=True)
        return annotations

    def load_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Adjust based on your image types
                    image_paths.append(os.path.join(root, file))
        return image_paths

    
def prepare_dataset(num_partitions: int = 20, batch_size: int = 256, val_ratio: float = 0.1):
    img_dir = r"D:\DACN\flower_federated\data\HELMET_DATASET\image"
    annotations_dir = r"D:\DACN\flower_federated\data\HELMET_DATASET\annotation"

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    dataset = HelmetDataset(img_dir=img_dir, annotations_dir=annotations_dir, transform=transform)

    total_length = len(dataset)
    partition_lengths = [total_length // num_partitions] * num_partitions

    remainder = total_length % num_partitions
    for i in range(remainder):
        partition_lengths[i] += 1

    partitions = random_split(dataset, partition_lengths, torch.Generator().manual_seed(2023))

    trainloaders = []
    valloaders = []

    for partition in partitions:
        num_total = len(partition)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(partition, [num_train, num_val], torch.Generator().manual_seed(2023))
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    testloader = DataLoader(dataset, batch_size=256)

    return trainloaders, valloaders, testloader

trainloaders, valloaders, testloader = prepare_dataset()
