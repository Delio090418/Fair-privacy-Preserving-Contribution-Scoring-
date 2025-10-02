import torch
import numpy as np
import torch.nn as nn
import math
import torchvision
# from torchvision import transforms
# from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import random_split,DataLoader, Subset, TensorDataset,Dataset
import random
from collections import defaultdict
import pathlib
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from data_paths import data_paths


np.random.seed(42)
torch.manual_seed(42)


datamnist=data_paths["mnist"]
#transform and path for brain data set
datacifar10=data_paths["cifar10"]

source_dir= data_paths["BRAIN"]
source_dir = pathlib.Path(source_dir)

##transfor for brain data set
transform_brain = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_mnist= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def testing_training(data_name):
    if data_name=="mnist":
        dataset=MNIST(root=datamnist, train=True, download=False, transform=transform_mnist)
    elif data_name=="cifar10":
        dataset = CIFAR10(root=datacifar10, train=True, download=False, transform=transform_cifar10)
    elif data_name=="BRAIN":
        dataset = torchvision.datasets.ImageFolder(source_dir, transform=transform_brain)
    else:
        raise ValueError("Not support data set")

    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # 20% for validation

# Split dataset into training and validation
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train=set_to_dataset(dataset, train_dataset)
    test=set_to_dataset(dataset, test_dataset)
    return train, test



class set_to_dataset(Dataset):
    def __init__(self, dataset, subset):
        self.dataset = dataset
        self.indices = subset.indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, label

# def subset_to_tensordataset(original_dataset, subset):
#     """
#     Convert a Subset to a TensorDataset.
    
#     Args:
#         original_dataset (Dataset): The original dataset (e.g., CIFAR-10).
#         subset (Subset): A subset of the original dataset.
        
#     Returns:
#         TensorDataset: A TensorDataset containing data and labels from the subset.
#     """
#     indices = subset.indices
#     # Get the data and targets using the subset indices
#     data = torch.stack([original_dataset[i][0] for i in indices])  # Collect all data in the subset
#     targets = torch.tensor([original_dataset[i][1] for i in indices])  # Collect all labels in the subset
    
#     # Create a TensorDataset from the data and targets
#     return TensorDataset(data, targets)




class RandomizedResponseDataset(Dataset):
    """
    Custom dataset wrapper that applies the Randomized Response (RR) mechanism to the labels.
    
    Args:
        dataset (Dataset): Original dataset (e.g., CIFAR-10).
        noise_rate (float): The probability of noise (e.g., 0.5 means 50% chance).
    """
    def __init__(self, dataset, noise_rate=0.5):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_classes = 4  # Number of classes in the dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Apply the randomized response noise to the label
        noisy_label = self._apply_randomized_response(label)

        return image, noisy_label

    def _apply_randomized_response(self, label):
        """
        Applies the Randomized Response (RR) mechanism to the label.

        Args:
            label (int): Original label of the image.

        Returns:
            noisy_label (int): The noisy label after applying RR.
        """
        if random.random() < self.noise_rate:
            # With probability (1 - noise_rate), the label is kept unchanged
            if random.random() < self.noise_rate/(self.num_classes-1): 
                new_label = label
                while new_label == label:
                    new_label = random.choice(range(self.num_classes))
                return new_label
            else:
                # Otherwise, choose a random incorrect label
                return label
        else:
            # No noise, return the original label
            return label


def partition_data(dataname, num_clients, alpha=0.5, partition_type='N-IID',seed=42):
    train= testing_training(dataname)[0]
    np.random.seed(seed)
    if dataname=="BRAIN" or dataname=="mnist" or dataname=="cifar10":
        targets = []
        for _, target in train:
            targets.append(target)
        labels = np.array(targets)
        num_samples = len(labels)
        num_classes = len(np.unique(labels))
        client_indices = defaultdict(list)
        
        if partition_type == 'N-IID':
            indices_per_class = {c: np.where(labels == c)[0] for c in range(num_classes)}
            for c in range(num_classes):
                np.random.shuffle(indices_per_class[c])
                num_samples_class = len(indices_per_class[c])
                proportions = np.random.dirichlet((alpha)* np.ones(num_clients))#alpha+0.2
                proportions = (np.array(proportions) * num_samples_class).astype(int)
                while proportions.sum() < num_samples_class:
                    proportions[np.argmax(proportions)] += 1
                while proportions.sum() > num_samples_class:
                    proportions[np.argmax(proportions)] -= 1
                start = 0
                for i in range(num_clients):
                    end = start + proportions[i]
                    client_indices[i].extend(indices_per_class[c][start:end])
                    start = end
        elif partition_type == 'IID':
            # IID partitioning: Shuffle and equally distribute data
            indices = np.random.permutation(num_samples)
            samples_per_client = num_samples // num_clients
            for i in range(num_clients):
                client_indices[i] = indices[i * samples_per_client: (i + 1) * samples_per_client].tolist()
    else:
        raise ValueError("Not support data set")
    return client_indices


def data_clients(data_name, num_clients, alpha=0.5, partition_type='N-IID'):
    train= testing_training(data_name)[0]
    client_partitions= partition_data(data_name, num_clients, alpha, partition_type=partition_type)
    client_data = {}
    
    if partition_type=="N-IID":
        for client, indices in client_partitions.items():
            client_datos=Subset(train,indices)
            cl_datos=set_to_dataset(train,client_datos)
            train_size_client = int(0.8 * len(cl_datos))  # 80% for training
            test_size_client = len(cl_datos) - train_size_client 
            entrenar, probar= random_split(cl_datos,[train_size_client,test_size_client])
            train_loader = DataLoader(entrenar,batch_size=64, shuffle=True,num_workers=0,
    pin_memory=True)
            test_loader = DataLoader(probar,batch_size=64, shuffle=False,num_workers=0,
    pin_memory=True)
            client_data[client] = {"train_loader": train_loader, "test_loader": test_loader}
    elif partition_type=="IID":
        noise_levels = [(i+1)/(num_clients+1) for i in range(num_clients)]
        for client, indices in client_partitions.items():
            client_datos=Subset(train,indices)
            cl_datos=set_to_dataset(train,client_datos)
            noisy_cliente=RandomizedResponseDataset(cl_datos,noise_rate=noise_levels[client])
            train_size_client = int(0.8 * len(noisy_cliente))  # 80% for training
            test_size_client = len(noisy_cliente) - train_size_client 
            entrenar, probar= random_split(noisy_cliente,[train_size_client,test_size_client])
            train_loader = DataLoader(entrenar,batch_size=64, shuffle=True,num_workers=0,
    pin_memory=True)
            test_loader = DataLoader(probar,batch_size=64, shuffle=False,num_workers=0,
    pin_memory=True)
            client_data[client] = {"train_loader": train_loader, "test_loader": test_loader}
    
    return client_data


def commun_test(data_name):
    test= testing_training(data_name)[1]
    test_loader = DataLoader(test, batch_size=64, shuffle=False,num_workers=0,
    pin_memory=True)
    return test_loader


if __name__ == "__main__":

    data_name="cifar10"
    partition="N-IID"
    alpha=.5#0.14
    num_clients=3
    set=data_clients(data_name,num_clients,alpha,partition)
    print(len(set[0]["test_loader"].dataset))
    print(len(set[1]["test_loader"].dataset))
    print(len(set[2]["test_loader"].dataset))
