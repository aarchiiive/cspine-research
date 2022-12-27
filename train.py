# torch library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# library
import time
import os
import copy
from tqdm import tqdm
import gc
import pandas as pd

# local library
from model import _models
from utils import *
from test import *

def train_model(model, 
                dataloaders, 
                criterion, 
                optimizer, 
                scheduler,
                device,
                save_path="", 
                num_epochs=25,
                batch_size=16,
                num_classes=4, 
                class_names=None,
                resume=False,
                use_wandb=False,
                start=0):
    
    since = time.time() # 시작 시간
    weights_path = os.path.join("weights", save_path) # weights를 저장할 경로
    best_acc = 0 # best accuracy
    max_acc = [0 for _ in range(num_classes)] # best accuracy per class
    
    # 폴더가 없다면 폴더 생성
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)
    
    if torch.cuda.device_count() > 1:
        print("Training with multiple devices")
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # model을 device에 올리기
    model = model.to(device)
    criterion = criterion.to(device)
    
    # wandb를 사용한다면 model의 모든 parameter를 watch
    if use_wandb:
        wandb.watch(model, log='all')
    
    # 학습 중 accuracy, loss를 저장할 dict
    learning_dict = {"train" : [],
                     "val" : []}
    learning_columns = ["epoch"] \
                     + get_columns(num_classes) \
                     + ["best", "accuracy", "loss"]
    
    # 만약 학습을 재개한다면 데이터를 불러옴
    if resume or os.path.isfile(os.path.join(weights_path, "train.csv")):
        print("Restart training model saved in {}".format(save_path))
        model.load_state_dict(torch.load(os.path.join(weights_path, "last.pt")))
        
        _train = pd.read_csv(os.path.join(weights_path, "train.csv"))
        _val = pd.read_csv(os.path.join(weights_path, "val.csv"))
        # df = {"train" : _train, "val" : _val}
        for t, v in zip(_train.values, _val.values):
            learning_dict["train"].append(t)
            learning_dict["val"].append(v)

        best_acc = int(_val["best"].tolist()[-1])
        for i in range(num_classes):
            max_acc[i] = _val["acc_{}".format(i)].max()
        start = len(_train)
        
    # class_names이 없다면 [0, 1, 2...]의 형태로
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    try:
        for epoch in range(start, num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 120)
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train() # train 단계일 경우
                else:
                    model.eval() # valid 단계일 경우

                running_loss = running_corrects = 0 # loss, corrects (오차, 맞은 개수)
                epoch_labels = epoch_probs = epoch_preds = None # labels, probs, preds (labels, 확률, 예측값)
                
                # 메모리 최적화
                gc.collect()
                torch.cuda.empty_cache()

                for inputs, labels in tqdm(dataloaders[phase]):
                    # tensor를 device로(cuda, cpu 등..)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):  
                        outputs = model(inputs) # 학습 시킬 사진들(batch_size만큼)을 모델에 넣고 outputs를 반환
                        _, preds = torch.max(outputs, 1) # label과 같은 정수 형태로 반환(0 or 1 or 2....)
                        loss = criterion(outputs, labels.long()) # loss 계산

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        
                    running_loss += loss.item() * inputs.size(0) # loss 계산
                    running_corrects += torch.sum(preds == labels.data) # 정답이 맞은 개수 계산
                    
                    # 성능 검증을 위해 사용할 labels, probs, preds 저장
                    if epoch_labels is None:
                        epoch_labels = labels
                        epoch_probs = outputs
                        epoch_preds = preds
                    else:
                        epoch_labels = torch.cat((epoch_labels, labels),dim=0)
                        epoch_probs = torch.cat((epoch_probs, outputs),dim=0)
                        epoch_preds = torch.cat((epoch_preds, preds),dim=0)
                
                print("\n{}".format(phase))
                row = [epoch]
                
                # 각 단계의 데이터셋에서 class별 데이터 개수를 계산
                _, total_counts = torch.unique(epoch_labels, return_counts=True)
                
                # class별 accuracy, loss 계산
                for i in range(num_classes):
                    acc = loss = 0 
                    for prob, pred, label in zip(epoch_probs, epoch_preds, epoch_labels):
                        if label == i:
                            loss += criterion(prob.reshape(1, -1), label.reshape(1).long())
                            if pred == i: acc += 1

                    acc = acc / total_counts[i] * 100
                    loss = loss / total_counts[i]
                    
                    # best accuracy 계산
                    if acc > max_acc[i] and phase == "val":
                        max_acc[i] = acc
                        if num_classes > 2:
                            torch.save(model.state_dict(), os.path.join(weights_path, "best_{}.pt".format(i)))
                        
                    if phase == "train": print("[{}] Accuracy : {:.6f} | Loss : {:.6f}".format(class_names[i], acc, loss))
                    elif phase == "val": print("[{}] Accuracy : {:.6f} | Loss : {:.6f} | Best : {:.6f}".format(class_names[i], acc, loss, max_acc[i]))
                    row.append(acc.item())
                    row.append(loss.item())
                            
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) * 100
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                
                print()
                
                if phase == "train":
                    scheduler.step()
                    print("Total accuracy : {:.6f}%".format(epoch_acc))
                    print("Total loss     : {:.6f}".format(epoch_loss))
                    
                elif phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc.item()
                        torch.save(model.state_dict(), os.path.join(weights_path, "best.pt"))
                        
                    if use_wandb:
                        # wandb에 pr curve, roc curve를 저장
                        wandb.log({"pr" : wandb.plot.pr_curve(epoch_labels.cpu(), epoch_probs.cpu(), labels=class_names, classes_to_plot=None)})
                        wandb.log({"roc" : wandb.plot.roc_curve(epoch_labels.cpu(), epoch_probs.cpu(), labels=class_names, classes_to_plot=None)})
                        
                    print("Total accuracy : {:.6f}%".format(epoch_acc))
                    print("Best accuracy  : {:.6f}%".format(best_acc))
                    print("Total loss     : {:.6f}".format(epoch_loss))
                    
                row.append(best_acc)
                row.append(epoch_acc.item())
                row.append(epoch_loss)
                learning_dict[phase].append(row)
                    
            # 현재까지 학습 시킨 weights를 저장
            torch.save(model.state_dict(), os.path.join(weights_path, "last.pt"))
            train_dict = pd.DataFrame(learning_dict["train"], columns=learning_columns)
            val_dict = pd.DataFrame(learning_dict["val"], columns=learning_columns)
            
            try:
                # 성능 지표들을 저장
                train_dict.to_csv(os.path.join(weights_path, "train.csv"), index=False)
                val_dict.to_csv(os.path.join(weights_path, "val.csv"), index=False)
            except:
                print("Failed saving log files : permission denied")
                
            if use_wandb:
                log_to_wandb(train_dict, val_dict, num_classes) # wandb에 기록
                
            print()
            
    except KeyboardInterrupt:
        # ctrl + C를 누를 경우 학습 내용을 저장하고 종료
        time_elapsed = time.time() - since
        print(f'Training stopped in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        torch.save(model.state_dict(), os.path.join(weights_path, "last.pt"))
        wandb.finish(1, quiet=True)
        
    except torch.cuda.OutOfMemoryError:
        wandb.finish(1, quiet=True)
        print("Error : torch.cuda.OutOfMemoryError")
        exit()
    except:
        time_elapsed = time.time() - since
        print(f'Training stopped in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        torch.save(model.state_dict(), os.path.join(weights_path, "last.pt"))
        wandb.finish(1, quiet=True)
        exit()

    time_elapsed = time.time() - since
    
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    learning_dict["train"] = pd.DataFrame(learning_dict["train"], columns=learning_columns)
    learning_dict["val"] = pd.DataFrame(learning_dict["val"], columns=learning_columns)
    learning_dict["train"].to_csv(os.path.join(weights_path, "train.csv"), index=False)
    learning_dict["val"].to_csv(os.path.join(weights_path, "val.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(weights_path, "last.pt"))
    wandb.finish(1, quiet=True)
    
def train(dataset_path,
          save_path,
          model_name,
          num_epochs=25,
          input_size=512,
          class_names=["normal", "abnormal"],
          optimize="adam",
          learning_rate=0.0001,
          weight_decay=0.0005,
          drop_rate=0.2,
          batch_size=8,
          num_workers=10,
          resume=False,
          foramen=0,
          use_wandb=True,
          project_name=None,
          start=0):
    
    # class 개수
    num_classes = len(class_names)
    # 가중치를 저장할 폴더 생성
    if not os.path.isdir("weights"):
        os.mkdir("weights")
    if not os.path.isdir(os.path.join("weights", save_path)):
        os.mkdir(os.path.join("weights", save_path))
    if use_wandb and project_name == None:
        # wandb를 시작하려고 하는데 project name이 없을 경우
        print("Set project name for using wandb")
        exit()
    
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # device = torch.device("mps")
    
    print("save weights in {}.....".format(save_path))
    print("save hyperparmeters.....")
    
    # hyperparmeters 저장하기 + wandb init
    save_params(os.path.join("weights", save_path),
                save_path,
                model_name,
                num_epochs,
                input_size,
                num_classes,
                optimize,
                learning_rate,
                weight_decay,
                drop_rate,
                batch_size,
                num_workers,
                use_wandb,
                project_name)
    
    # foramen 별로 레이블 고르기
    train_label, val_label = select_labels(foramen)
    print("select labels.....")
    print(train_label)
    print(val_label)
    
    # dataloaders
    trainloader, testloader = dataloader(os.path.join(dataset_path, "train"),
                                         os.path.join(dataset_path, "val"),
                                        [os.path.join(dataset_path, train_label),
                                         os.path.join(dataset_path, val_label)],
                                         input_size=input_size,
                                         batch_size=batch_size,
                                         num_workers=num_workers)

    dataloaders = {'train': trainloader, 
                    'val' : testloader}
        
    # model 불러오기
    model = _models(model_name=model_name, 
                    num_classes=num_classes,
                    drop_rate=drop_rate)
    
    # criterion
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    df = pd.read_csv(os.path.join(dataset_path, "label/train.csv"))
    weights = compute_class_weight(class_weight = "balanced", classes=np.array(range(num_classes)), y = df["label"])
    weights = torch.FloatTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=weights)
 
    # opimizer
    # https://pytorch.org/docs/stable/optim.html
    if optimize=="adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    elif optimize=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # train model
    train_model(model,
                dataloaders,
                criterion,
                optimizer,
                scheduler,
                device,
                save_path,
                num_epochs,
                batch_size,
                num_classes,
                class_names,
                resume=resume,
                use_wandb=use_wandb,
                start=start)

    wandb.finish(1, quiet=True)