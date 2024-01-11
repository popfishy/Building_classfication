# -*- coding:utf-8 -*-
# @Time : 2024-01-08 21:56
# @Author: popfishy
# @File : train.py

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import torch.utils.data as tud
import numpy as np
import matplotlib.pyplot as plt
from model.inception_resnet_v2 import Inception_ResNetv2
import albumentations as A
from model.vit import ViT
from torch.utils.tensorboard import SummaryWriter


dataset_root = "/home/yjq/dataset_augmentation"
global_model_name = "Vit"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("logs")

batch_size = 32
input_size = 299
num_class = 21

f = open("result_" + global_model_name + ".txt", "w")

dataset = torchvision.datasets.ImageFolder(
    root=dataset_root,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomAffine(
                degrees=(-5, 5), translate=(0.08, 0.08), scale=(0.9, 1.1)
            ),
            torchvision.transforms.Resize(300),  # 调整图像短边
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
        ]
    ),
)

print(dataset, "\n")
print("classes:\n", dataset.classes, "\n")

# Split dataset into train and test (7:3)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [int(len(dataset) * 0.7), len(dataset) - int(len(dataset) * 0.7)]
)
train_dataloader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = tud.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def initialize_model(model_name, num_class, use_pretrained, feature_extract):
    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        if feature_extract:  # do not update the parameters
            for param in model_ft.parameters():
                param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_class)
    else:
        print("model not implemented")
        return None
    return model_ft


def train_model(model, train_dataloader, loss_fn, optimizer, epoch):
    model = model.to(device)
    model.train()
    total_loss = 0.0
    total_corrects = 0.0
    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * inputs.size(0)
        total_corrects += torch.sum(preds.eq(labels))
    epoch_loss = total_loss / len(train_dataloader.dataset)
    epoch_accuracy = total_corrects / len(train_dataloader.dataset)
    f.write(
        "Epoch:{}, Training Loss:{}, Traning Acc:{}\n".format(
            epoch, epoch_loss, epoch_accuracy
        )
    )
    print(
        "Epoch:{}, Training Loss:{}, Traning Acc:{}\n".format(
            epoch, epoch_loss, epoch_accuracy
        )
    )
    writer.add_scalar("Loss/train", epoch, epoch_loss)
    writer.add_scalar("Accuracy/train", epoch, epoch_accuracy)


def test_model(model, test_dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_corrects = 0.0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds.eq(labels))
    epoch_loss = total_loss / len(test_dataloader.dataset)
    epoch_accuracy = total_corrects / len(test_dataloader.dataset)
    f.write("Test Loss:{}, Test Acc:{}\n".format(epoch_loss, epoch_accuracy))
    print("Test Loss:{}, Test Acc:{}\n".format(epoch_loss, epoch_accuracy))
    writer.add_scalar("Loss/test", epoch, epoch_loss)
    writer.add_scalar("Accuracy/test", epoch, epoch_accuracy)
    return epoch_accuracy


# TODO start train
# model = initialize_model(global_model_name, 20, use_pretrained=True, feature_extract=True)
model = Inception_ResNetv2()
model.load_state_dict(torch.load("results/Inception_ResNetv2_50_2.pth"))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
num_epochs = 50
best_epoch = 0
best_acc = 0.95
test_accuracy_hist = []
for epoch in range(num_epochs):
    train_model(model, train_dataloader, loss_fn, optimizer, epoch)
    acc = test_model(model, test_dataloader, loss_fn)
    test_accuracy_hist.append(acc.item())
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
        writer.add_histogram(f"{name}.grad", param.grad, epoch)
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        torch.save(model.state_dict(), "best.pth")
    if (epoch + 1) % 10 == 0:
        torch.save(
            model.state_dict(), global_model_name + "_" + str(epoch + 1) + ".pth"
        )
f.close()
writer.close()

torch.save(model.state_dict(), global_model_name + "_" + str(epoch + 1) + ".pth")

plt.figure(1)
plt.title("Test Accuracy vs. Training Epoch")
plt.xlabel("Training Epochs")
plt.ylabel("Test Accuracy")
plt.plot(range(1, num_epochs + 1), test_accuracy_hist, label="Pretrained")
plt.ylim((0, 1.0))
plt.xticks(np.arange(1, num_epochs + 1, 1.0))
plt.legend()
plt.show()
plt.savefig(1, "pic/Loss.png")
