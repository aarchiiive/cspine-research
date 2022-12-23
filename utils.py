import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import gc
import numpy as np
import pandas as pd
from PIL import Image
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_columns(num_classes):
    temp = []
    for i in range(num_classes):
        for l in ["acc_", "loss_"]:
            temp.append(l + str(i))
    return temp

def log_to_wandb(train_dict, val_dict, num_classes):
    """
    wandb에 지표들을 기록

    Args:
        train_dict (dict): train 단계의 성능 지표
        val_dict (dict): valid 단계의 성능 지표
        num_classes (int): class 개수
    """
    # train
    table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(train_dict["epoch"].tolist(), 
                                  train_dict["accuracy"].tolist())], \
                                  columns = ["epoch", "accuracy"])
    wandb.log({"train_accuracy" : wandb.plot.line(table, "epoch", "accuracy", title="train/accuracy")})
    table = wandb.Table(data=[[x, y] for (x, y) in \
                            zip(train_dict["epoch"].tolist(), 
                                train_dict["loss"].tolist())], \
                                columns = ["epoch", "loss"])
    wandb.log({"train_loss" : wandb.plot.line(table, "epoch", "loss", title="train/loss")})
    # valid
    table = wandb.Table(data=[[x, y] for (x, y) in \
                              zip(val_dict["epoch"].tolist(), 
                                  val_dict["accuracy"].tolist())], \
                                  columns = ["epoch", "accuracy"])
    wandb.log({"valid_accuracy" : wandb.plot.line(table, "epoch", "accuracy", title="valid/accuracy")})
    table = wandb.Table(data=[[x, y] for (x, y) in \
                            zip(val_dict["epoch"].tolist(), 
                                val_dict["loss"].tolist())], \
                                columns = ["epoch", "loss"])
    wandb.log({"valid_loss" : wandb.plot.line(table, "epoch", "loss", title="valid/loss")})
    
    for i in range(num_classes):
        # train
        table = wandb.Table(data=[[x, y] for (x, y) in \
                            zip(train_dict["epoch"].tolist(), 
                                train_dict["acc_{}".format(i)].tolist())], \
                                columns = ["epoch", "accuracy"])
        wandb.log({"train_acc_{}".format(i) : wandb.plot.line(table, "epoch", "accuracy", title="train/accuracy/class {}".format(i))})
        table = wandb.Table(data=[[x, y] for (x, y) in \
                            zip(train_dict["epoch"].tolist(), 
                                train_dict["loss_{}".format(i)].tolist())], \
                                columns = ["epoch", "loss"])
        wandb.log({"train_loss_{}".format(i) : wandb.plot.line(table, "epoch", "loss", title="train/loss/class {}".format(i))})
        # valid
        table = wandb.Table(data=[[x, y] for (x, y) in \
                            zip(val_dict["epoch"].tolist(), 
                                val_dict["acc_{}".format(i)].tolist())], \
                                columns = ["epoch", "accuracy"])
        wandb.log({"valid_acc_{}".format(i) : wandb.plot.line(table, "epoch", "accuracy", title="valid/accuracy/class {}".format(i))})
        table = wandb.Table(data=[[x, y] for (x, y) in \
                            zip(val_dict["epoch"].tolist(), 
                                val_dict["loss_{}".format(i)].tolist())], \
                                columns = ["epoch", "loss"])
        wandb.log({"valid_loss_{}".format(i) : wandb.plot.line(table, "epoch", "loss", title="valid/loss/class {}".format(i))})

def save_params(weights_path,
                save_path,
                model_name,
                num_epochs=25,
                input_size=600,
                num_classes=4,
                optimizer="adam",
                learning_rate=0.0001,
                weight_decay=0.0005,
                drop_rate=0.2,
                batch_size=8,
                num_workers=10,
                use_wandb=False,
                project_name=None):
    """
    한 학습에서 사용되는 hyperparameter들을 저장하고 wandb를 시작

    Args:
        weights_path (str): 가중치를 저장하는 경로
        save_path (str): 가중치를 저장하는 폴더 이름
        model_name (str): 사용한 모델 이름
        num_epochs (int, optional): epochs(학습 횟수). Defaults to 25.
        input_size (int, optional): image 크기. Defaults to 600.
        num_classes (int, optional): class의 개수. Defaults to 4.
        optimizer (str, optional): optimizer. Defaults to "adam".
        learning_rate (float, optional): learning rate. Defaults to 0.0001.
        weight_decay (float, optional): weight decay. Defaults to 0.0005.
        drop_rate (float, optional): probability of dropout. Defaults to 0.2.
        batch_size (int, optional): batch size. Defaults to 8.
        num_workers (int, optional): 사용할 cpu 코어 개수. Defaults to 10.
        use_wandb (bool, optional): wandb 사용여부. Defaults to False.
        project_name (_type_, optional): wandb project name. Defaults to None.
    """
    
    if use_wandb:
        wandb.init(project=project_name, name=save_path+"-"+model_name, tags=[model_name], group="train")

        # if os.path.isfile(os.path.join(weights_path, "train.csv")):
        #     wandb.init(project="skin-research", name=save_path+"-"+model_name, resume=True)
        # else:
        #     wandb.init(project="skin-research", name=save_path+"-"+model_name)
            
        wandb.config = {
            "model_name" : model_name,
            "num_epochs" : num_epochs,
            "input_size" : input_size,
            "num_classes" : num_classes,
            "optimizer" : optimizer,
            "learning_rate" : learning_rate,
            "weight_decay" : weight_decay,
            "drop_rate" : drop_rate,
            "batch_size" : batch_size,
            "num_workers" : num_workers
        }
    
    df = pd.DataFrame.from_dict({
        "model_name" : [model_name],
        "num_epochs" : [num_epochs],
        "input_size" : [input_size],
        "num_classes" : [num_classes],
        "optimizer" : [optimizer],
        "learning_rate" : [learning_rate],
        "weight_decay" : [weight_decay],
        "drop_rate" : [drop_rate],
        "batch_size" : [batch_size],
        "num_workers" : [num_workers]
    }, orient="columns")
    
    df.to_csv(os.path.join(weights_path, "hyperparameters.csv"))


def dataloader(train_path,
               test_path,
               label_path,
               input_size,
               batch_size, 
               num_workers):
    """
    dataloader : 학습 시 image를 불러오는 기능을 함

    Args:
        train_path (str): train dataset 경로
        test_path (str): test dataset 경로
        label_path (str): label 경로 -> [../train.csv, ../val.csv]
        input_size (int): image 크기
        batch_size (int): batch size
        num_workers (int): 사용할 cpu 코어 개수

    Returns:
        _type_: _description_
    """
    
    train_label_path, test_label_path = label_path
    
    # image augentations
    # https://pytorch.org/vision/stable/transforms.html
    train_transform = transforms.Compose([
        # transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.1)),
        transforms.RandomResizedCrop(size=(input_size, input_size), scale=(0.75, 1.0)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomErasing(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        # transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # train, valid 데이터셋
    train_data = CustomImageDataset(train_label_path, train_path, train_transform)
    test_data = CustomImageDataset(test_label_path, test_path, test_transform)

    print("Initializing Datasets and Dataloaders...")

    # 학습 시 쓰일 dataloader
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_dataloader, test_dataloader

def get_testloader(image_path, 
                   label_path, 
                   input_size=512,
                   batch_size=8):
    """
    test 시 사용될 dataloader

    Args:
        image_path (str): image 폴더 경로
        label_path (str): label 경로
        input_size (int, optional): image 크기. Defaults to 512.
        batch_size (int, optional): batch size. Defaults to 8.

    Returns:
        dataloader: dataloder for test
    """
    
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_data = CustomImageDataset(label_path, image_path, test_transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_dataloader

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir=None, transform=None, target_transform=None, phase=None):
        """
        Custom dataset
        reference : https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        Args:
            annotations_file (str): _description_
            img_dir (str, optional): _description_. Defaults to None.
            transform (transform, optional): _description_. Defaults to None.
            target_transform (transform, optional): _description_. Defaults to None.
            phase (str, optional): _description_. Defaults to None.
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.phase = phase

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1]+".jpg")
        image = Image.open(img_path) # image 불러오기
        
        if self.transform: 
            image = self.transform(image) # augmentation

        if self.target_transform:
            label = self.target_transform(label)
            
        image = torch.FloatTensor(image)
        label = self.img_labels.iloc[idx, 2] # label 불러오기

        return image, label
    
    def __len__(self): # dataset의 전체 크기(= image의 개수) 반환
        return len(os.listdir(self.img_dir))

def predictions(model, test_loader, device):
    # 예측값 반환 -> test시 사용
    total_probs = None
    total_preds = None
    true_labels = None
    gc.collect()
    torch.cuda.empty_cache()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            true_labels = labels if true_labels is None else torch.cat((true_labels, labels),dim=0)
            total_probs = outputs if total_probs is None else torch.cat((total_probs, outputs),dim=0)
            total_preds = preds if total_preds is None else torch.cat((total_preds, preds),dim=0)

    return total_probs.cpu(), total_preds.cpu(), true_labels.cpu().numpy()

def confidence_interval(auc, labels, z=1.959964):
    # AUC의 95% 신뢰구간 
    _, counts = np.unique(labels, return_counts=True)
    n1, n2 = counts
    q0 = auc * (1 - auc)
    q1 = auc / (2 - auc) - auc**2
    q2 = 2 * auc**2 / (1 + auc) - auc**2
    se = np.sqrt((q0 + (n1 - 1)*q1 + (n2 - 1)*q2) / (n1 * n2))
    return [auc - z*se, auc + z*se]
    
if __name__ == "__main__":
    pass