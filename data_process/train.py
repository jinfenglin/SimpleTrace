import engine
import config
from model import BertTrace, Models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import dataset
from collections import defaultdict
import logging
import os
from transformers import AutoConfig

transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)


def run():
    df = pd.read_csv(config.TRAINING_FILE).fillna("None")
    df_train, df_valid = train_test_split(
        df, test_size=0.4, random_state=42, stratify=df.labels.values
    )
    print("shape of the training dataset is {}".format(len(df_train)))

    df_valid, df_test = train_test_split(
        df_valid,
        test_size=0.1,
        random_state=42,
    )

    train_data_loader = dataset.create_data_loader(
        df_train, config.TRAIN_BATCH_SIZE, num_workers=4
    )
    valid_data_loader = dataset.create_data_loader(
        df_valid, config.VALID_BATCH_SIZE, num_workers=2
    )
    test_data_loader = dataset.create_data_loader(
        df_test, config.VALID_BATCH_SIZE, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supported_models = [config.PRE_TRAINED_MODEL]

    for mdl in supported_models:
        print("Working on {} model".format(mdl))
        models = Models(mdl)

        # models = BertTrace.from_pretrained(mdl, config=AutoConfig.from_pretrained(mdl))
        models.to(device)

        optimizer = AdamW(models.parameters(), lr=5e-5, correct_bias=False)

        num_train_steps = (
            len(train_data_loader) * config.EPOCHS
        )  # len(df_train)/len(train_data_loader) * EPOCHS

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )

        history = defaultdict(list)
        best_acc = 0
        for epoch in range(config.EPOCHS):
            dir_path = os.path.join(os.getcwd(), mdl[0:4])

            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            path = os.path.join(dir_path, "model.pth")
            print(path)

            print(f"Epoch {epoch + 1}/{config.EPOCHS}")
            print("-" * 10)

            train_acc, train_loss = engine.train_epoch(
                train_data_loader, models, optimizer, scheduler, device, len(df_train)
            )
            print(f"Train loss {train_loss} accuracy {train_acc}")

        return
        engine.trainingvsvalid(history["train_acc"], history["val_acc"], mdl)

        test_acc, _ = engine.eval_epoch(test_data_loader, models, device, len(df_test))
        print("The test accuracy of the {} is {}".format(mdl, test_acc.item()))

        y_review_texts, y_pred, y_pred_probs, y_test = engine.get_prediction(
            test_data_loader, models, device, len(df_test)
        )

        class_names = ["0", "1"]
        print("Classification report for the test dataset on {} model".format(mdl))
        print(classification_report(y_test, y_pred, target_names=class_names))

        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        engine.show_confusion_matrix(df_cm)

        break


if __name__ == "__main__":
    run()
