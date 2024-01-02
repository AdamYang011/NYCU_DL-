import os 
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from torchinfo import summary

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Raw dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_labels = {}

        # Create a mapping of class labels to integers
        self.class_labels = {}
        class_idx = 0

        # Iterate over sub-directories
        for class_dir in sorted(os.listdir(self.root_dir)):
            class_dir_path = os.path.join(self.root_dir, class_dir)
            print(class_dir_path)
            if os.path.isdir(class_dir_path):
                self.class_labels[class_dir] = class_idx
                class_idx += 1

                # Iterate over images in the sub-directory
                for img_filename in sorted(os.listdir(class_dir_path)):
                    if img_filename.endswith(".jpg"):
                        img_path = os.path.join(class_dir_path, img_filename)
                        self.images.append(img_path)
                        self.labels.append(self.class_labels[class_dir])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])        
        label = self.labels[idx]
        
        # Check for grayscale images and convert to RGB
        if image.mode == "L":
            image = Image.merge("RGB", (image, image, image))

        if self.transform:
            image = self.transform(image)
            
        return image, label

# Test data have different data structure from train dataset
class ImageDatasetTest(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        # Iterate over images in the test sub-directory
        for img_filename in sorted(os.listdir(self.root_dir)):
            if img_filename.endswith(".jpg"):
                img_path = os.path.join(self.root_dir, img_filename)
                self.images.append(img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])

        # Placeholder label for test set, because test dataset without label
        label = -1  

        # Check for grayscale images and convert to RGB
        if image.mode == "L":
            image = Image.merge("RGB", (image, image, image))

        # Apply transform if available
        if self.transform:
            image = self.transform(image)

        return image, label


# pytorch dataloader
def model_dataloder(weights, transform):
    weights = weights
    data_folder = "./achieve"
    train_folder = data_folder + "/train"
    val_folder = data_folder + "/valid"
    test_folder = data_folder + "/test"
    
    # pytorch dataset
    train_dataset = ImageDataset(train_folder, transform = transform)
    val_dataset = ImageDataset(val_folder, transform = transform)
    test_dataset = ImageDatasetTest(test_folder, transform=transform)

    # Check the length of each dataset
    #print(f"Train dataset length: {len(train_dataset)}")
    #print(f"Validation dataset length: {len(val_dataset)}")
    #print(f"Test dataset length: {len(test_dataset)}")
    
    # pytorch dataloader
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = 32, shuffle = False)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = 32, shuffle = False)
    
    return train_dataloader, val_dataloader, test_dataloader

# Train -> train_loss, train_acc
def train (model, dataloader, loss_fn, optimizer, device):
    train_loss, train_acc = 0, 0
    
    model.to(device)
    model.train()
    
    for batch, (x, y) in enumerate (dataloader):
        x, y = x.to(device), y.to(device)
        
        train_pred = model(x)
        
        loss = loss_fn(train_pred, y)
        train_loss = train_loss + loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_pred_label = torch.argmax(torch.softmax(train_pred, dim = 1), dim = 1)
        train_acc = train_acc + (train_pred_label == y).sum().item() / len(train_pred)
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

# Val -> val_loss, val_acc
def val (model, dataloader, loss_fn, device):
    val_loss, val_acc = 0, 0
    
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            val_pred = model(x)
            
            loss = loss_fn(val_pred, y)
            val_loss = val_loss + loss.item()
            
            val_pred_label = torch.argmax(torch.softmax(val_pred, dim = 1), dim = 1)
            val_acc = val_acc + (val_pred_label == y).sum().item() / len(val_pred)
        
        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)
        
        return val_loss, val_acc

# Training loop -> results dictionary
def training_loop(model, train_dataloader, val_dataloader, device, epochs, patience):
    # empty dict for restore results
    results = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    
    # hardcode loss_fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
    
    # loop through epochs
    for epoch in range(epochs):
        train_loss, train_acc = train(model = model, 
                                      dataloader = train_dataloader,
                                      loss_fn = loss_fn,
                                      optimizer = optimizer,
                                      device = device)
        
        val_loss, val_acc = val(model = model,
                                dataloader = val_dataloader,
                                loss_fn = loss_fn,
                                device = device)
        
        # print results for each epoch
        print(f"Epoch: {epoch+1}\n"
              f"Train loss: {train_loss:.4f} | Train accuracy: {(train_acc*100):.3f}%\n"
              f"Val loss: {val_loss:.4f} | Val accuracy: {(val_acc*100):.3f}%")
        
        # record results for each epoch
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        # calculate average "val_loss" for early_stopping
        mean_val_loss = np.mean(results["val_loss"])
        best_val_loss = float("inf")
        num_no_improvement = 0
        if np.mean(mean_val_loss > best_val_loss):
            best_val_loss = mean_val_loss
        
            model_state_dict = model.state_dict()
            best_model.load_state_dict(model_state_dict)
        else:
            num_no_improvement +=1
    
        if num_no_improvement == patience:
            break
    
    # plt results after early_stopping
    plt.figure(figsize = (8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(results["train_loss"], label = "Train loss")
    plt.plot(results["val_loss"], label = "Val loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(results["train_acc"], label = "Train accuracy")
    plt.plot(results["val_acc"], label = "Val accuracy")
    plt.legend()
    
    return results

if __name__ == '__main__':
    resnet_weight = torchvision.models.ResNet50_Weights.DEFAUL
    resnet_model = torchvision.models.resnet50(weights = resnet_weight)

    for param in resnet_model.parameters():
        param.requires_grad = False

    # Custom output layer
    # resnet_model.fc

    custom_fc = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(1000, 100))

    resnet_model.fc = nn.Sequential(
        resnet_model.fc,
        custom_fc
    )

    summary(resnet_model, input_size = (1, 3, 244, 244), col_names = ["output_size", "num_params", "trainable"], col_width = 15)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data augmentation
    # resnet_weight.transforms()
    resnet_transform = transforms.Compose([
        transforms.Resize(size = 232),
        transforms.ColorJitter(brightness = (0.8, 1.2)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(degrees = 15),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    resnet_train_dataloader, resnet_val_dataloader, resnet_test_dataloader = model_dataloder(weights = resnet_weight, 
                                                                                            transform = resnet_transform
                                                                                            )

    # Actual training ResNet model
    resnet_results = training_loop(model = resnet_model,
                                train_dataloader = resnet_train_dataloader,
                                val_dataloader = resnet_val_dataloader,
                                device = device,
                                epochs = 30,
                                patience = 5
                                )

    # empty list store predicted labels
    predict_label_list = []

    # eval mode
    resnet_model.eval()

    with torch.no_grad(): 
        for images, _ in resnet_test_dataloader:
            images = images.to(device)
            # Assuming batch size is 1 for simplicity
            logits = resnet_model(images)
            probabilities = torch.softmax(logits, dim=1)
            labels = torch.argmax(probabilities, dim=1).tolist()
            predict_label_list.extend(labels)

    # Get the CSV result
    df = pd.DataFrame({"ID": range(len(predict_label_list)), "Category": predict_label_list})
    df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")