import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import time
import copy
import numpy as np
import argparse

from resnet import resnet18,resnet34,snet

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform) 
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader

def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def train_kd(model, teacher, alpha, temperature, train_loader, test_loader, device):

    # The training configurations were not carefully selected.
    learning_rate = 1e-1
    num_epochs = 200

    criterion = nn.CrossEntropyLoss()
    kd_criterion = nn.KLDivLoss()
    T = temperature
    best_acc = 0

    model.to(device)
    if teacher:
        teacher.to(device)
        teacher.eval()
    else:
        alpha = 0

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    writer = SummaryWriter()

    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0
        running_kd_loss = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            s_loss = criterion(outputs, labels)
            
            if teacher:
                t_outputs = teacher(inputs)
                kd_loss = T * T * kd_criterion(nn.functional.log_softmax(outputs/T, dim=1), nn.functional.softmax(t_outputs/T, dim=1))
            else:
                kd_loss = torch.tensor([0]).to(device)
            
            
            loss = (1-alpha) * s_loss + alpha * kd_loss

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_kd_loss += kd_loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_kd_loss = running_kd_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion)
        best_acc = max(best_acc, eval_accuracy)
        
        scheduler.step()
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', eval_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', eval_accuracy, epoch)

        print("Epoch: {:02d} Train Loss: {:.3f} Train KD_Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.4f}".format(epoch, train_loss, train_kd_loss, train_accuracy, eval_loss, eval_accuracy))

    return model, best_acc

def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model

def create_model(model_arch, num_classes=10):

    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    # model = resnet18(num_classes=num_classes, pretrained=False)
    model_dic = {'res18':resnet18, 'res34':resnet34, 'snet':snet}
    model = model_dic[model_arch](num_classes=num_classes, pretrained=False)

    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Modify the last FC layer
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default='None')
    parser.add_argument('-s', type=str, required=True, default='res18')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=4.0)
    args = parser.parse_args()

    random_seed = 0
    num_classes = 10
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    teacher_arch = args.t
    student_arch = args.s
    model_dir = "saved_models"
    teacher_filename = teacher_arch + "_cifar10.pt"
    student_filename = student_arch + "_kd_" + teacher_arch + "_cifar10.pt"

    teacher_filepath = os.path.join(model_dir, teacher_filename)
    student_filepath = os.path.join(model_dir, student_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    if teacher_arch != 'None':
        teacher = create_model(teacher_arch, num_classes=num_classes)
        teacher = load_model(model=teacher, model_filepath=teacher_filepath, device=cuda_device)
    else:
        teacher = None
        
    student = create_model(student_arch, num_classes=num_classes)
    print(teacher)
    print(student)

    train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256)

    # Load a pretrained model.
    
    student,best_accuracy = train_kd(model=student, teacher=teacher, alpha=args.alpha, temperature=args.temperature, train_loader=train_loader, test_loader=test_loader, device=cuda_device)

    save_model(model=student, model_dir=model_dir, model_filename=student_filename)    

    print('Student best accuracy:',best_accuracy)
    


if __name__ == "__main__":

    main()
