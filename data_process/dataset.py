from transformers.models.auto.tokenization_auto import AutoTokenizer
import config
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers.tokenization_utils_base import EncodedInput
import pandas as pd
import numpy as np
from torch.utils.data.sampler import RandomSampler

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"


# BERTMODEL --- ROBERTA -- DISTILBERT -- ELECTRA


class ModelsDataset(Dataset):
    def __init__(self, S_text, T_text, labels):
        self.S_text = S_text
        self.T_text = T_text
        self.labels = labels
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.S_text)

    def __getitem__(self, item):
        S_text = str(self.S_text[item])
        T_text = str(self.T_text[item])
        labels = self.labels[item]
        tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL)
        encoding = tokenizer(
            S_text,
            T_text,
            add_special_tokens=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
            max_length=self.max_len,
        )

        return {
            "input_ids": encoding[INPUT_IDS].flatten(),
            "attention_mask": encoding[ATTENTION_MASK].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


def create_data_loader(df, batch_size, num_workers):
    ds = ModelsDataset(
        S_text=df.S_text.to_numpy(),
        T_text=df.T_text.to_numpy(),
        labels=df.labels.to_numpy(),
    )

    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, sampler=RandomSampler(ds)
    )


# df_train = pd.read_csv(config.TRAINING_FILE)
# # df_test = pd.read_csv(config.TEST_FILE)
# # df_val =  pd.read_csv(config.VAL_FILE)
# train_data_loader = create_data_loader(df_train)
# # test_data_loader = create_data_loader(df_test)
# # test_data_loader = create_data_loader(df_val)

# data = next(iter(train_data_loader))
# print(data.keys())
