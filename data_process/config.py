import transformers

MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
EPOCHS =10
ACCUMULATION =2 
MODEL_PATH = "model.bin"
TRAINING_FILE = "final_data.csv"
TEST_FILE = "../data/test.csv"
VAL_FILE = "../data/val.csv"
PRE_TRAINED_MODEL = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)