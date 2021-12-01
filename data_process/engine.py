"""training and evalutaion function"""
import numpy as np
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
import torch
import config
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns


def train_epoch(data_loader, model, optimizer, scheduler, device, n_examples):
    correct_pred = 0
    losses = []
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.train()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        labels = d["labels"].type(torch.LongTensor).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs.view(-1, 2), labels.reshape(-1))
        print(loss)

        y_pred = outputs.data.max(1)[1]
        cnt = y_pred.eq(labels).long().sum().item()
        print(cnt)
        correct_pred += cnt
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    return correct_pred / n_examples, np.mean(losses)


def eval_epoch(data_loader, model, device, n_examples):
    # model.eval()
    # correct_prediction = 0
    # losses = []
    # with torch.no_grad():
    #     for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
    #         input_ids = d["input_ids"].to(device)
    #         attention_mask = d["attention_mask"].to(device)
    #         labels = d["labels"].to(device)

    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    #         _, pred = torch.max(outputs, dim=1)
    #         loss = loss_fn(outputs, labels)
    #         correct_prediction += torch.sum(pred == labels)
    #         losses.append(loss.item())

    # return correct_prediction.double() / n_examples, np.mean(losses)
    return 0, 0


def get_prediction(data_loader, model, device, n_examples):
    model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    losses = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, pred = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            probs = F.softmax(outputs, dim=1)
            predictions.extend(pred)
            prediction_probs.extend(probs)
            real_values.extend(labels)
            losses.append(loss.item())

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

    # loss.backward()
    # optimizer.step()
    # scheduler.step()
    # optimizer.zero_grad()


def trainingvsvalid(train_acc, val_acc, model_name):
    plt.plot(train_acc, label="train accuracy")
    plt.plot(val_acc, label="validation accuracy")

    plt.title("Training history")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig("{}.png".format(model_name[0:4]))
    plt.show()


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True sentiment")
    plt.xlabel("Predicted sentiment")
