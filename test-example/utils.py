import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
from datasets import Dataset
import io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import wandb



def read_data():
    """
    return train , test
    """
    with open(os.path.join(os.path.abspath(os.getcwd()),"artifacts/detect_llm_raw_data:v1/train_df.table.json")) as json_data:
        data = json.load(json_data)
        train = pd.DataFrame(data = data["data"],columns=data["columns"])
        json_data.close()

    with open(os.path.expanduser(os.path.join(os.path.abspath(os.getcwd()),"artifacts/detect_llm_raw_data:v1/test_df.table.json"))) as json_data:
        data = json.load(json_data)
        test = pd.DataFrame(data = data["data"],columns=data["columns"])
        json_data.close()
    return train , test

def preprocess(train=None,test=None):
    """
    return dataset_train, dataset_test
    """
    train.fillna(" ",inplace=True)
    test.fillna(" ",inplace=True)
    train["text"] = train["Question"] + " " + train["Response"]
    test["text"] = test["Question"] + " " + test["Response"]
    df_train = train[["target","text"]]
    df_test = test[["text"]]
    dataset_train = Dataset.from_pandas(df_train)
    dataset_test = Dataset.from_pandas(df_test)
    
    return dataset_train, dataset_test


def dataset_tokenize_n_split(train, dataset_train, dataset_test,model_name):
    """
    return split_train_dataset,split_eval_dataset , tokenized_test , tokenizer
    """
    tokenizer       = AutoTokenizer.from_pretrained(model_name )
    def tokenize_function(examples):
    
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train = dataset_train.map(tokenize_function, batched=True)
    tokenized_test  = dataset_test.map(tokenize_function, batched=True)
    tokenized_train = tokenized_train.remove_columns(['text'])
    tokenized_train = tokenized_train.rename_column("target", "labels")
    tokenized_test = tokenized_test.remove_columns(['text'])

    kf= StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    for i , (tr_idx,val_idx) in enumerate(kf.split(train,train.target)):
        print(f"Fold : {i}")
        print(f"shape train : {tr_idx.shape}")
        print(f"shape val : {val_idx.shape}")
        break
        
    
    split_train_dataset = tokenized_train.select(tr_idx)
    split_eval_dataset = tokenized_train.select(val_idx)

    return split_train_dataset,split_eval_dataset , tokenized_test , tokenizer, train.iloc[val_idx]

def predict_fn(model,dataset_ = None):
    
    """
    return mean of all_probabilities (m,7)
    """
    input_ids = dataset_['input_ids']
    # token_type_ids = dataset_['token_type_ids']
    attention_mask = dataset_['attention_mask']

    # Move the input tensors to the GPU
    input_ids = torch.tensor(input_ids).to('cuda:0')
    # token_type_ids = torch.tensor(token_type_ids).to('cuda:0')
    attention_mask = torch.tensor(attention_mask).to('cuda:0')

    # Define batch size
    batch_size = 8

    # Calculate the number of batches
    num_samples = len(input_ids)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Initialize a list to store the softmax probabilities
    all_probabilities = []

    # Make predictions in batches
    with torch.no_grad():
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, num_samples)

            batch_input_ids = input_ids[start_idx:end_idx]
    #         batch_token_type_ids = token_type_ids[start_idx:end_idx]
            batch_attention_mask = attention_mask[start_idx:end_idx]

            outputs = model(input_ids=batch_input_ids, 
    #                         token_type_ids=batch_token_type_ids, 
                            attention_mask=batch_attention_mask)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=1)


            all_probabilities.extend(probabilities.tolist())
    return np.concatenate(all_probabilities,axis=0).reshape(dataset_.shape[0],7)


def conf_mat(df_val = None,preds_val = None):
    """
    no return
    """
    plt.figure(figsize=(8,8))
    ConfusionMatrixDisplay.from_predictions(df_val.target,np.argmax(preds_val,axis=1))
    plt.savefig(f"val_conf_matrix.png", format="png")
    plt.show();
    conf = wandb.Image(data_or_path="val_conf_matrix.png")
    wandb.log({"val_conf_matrix": conf})
def create_model(model_name = "distilroberta-base",num_labels = 7):
    """
    return
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # Specify the GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Move your model to the GPU
    model.to(device);
    
    return model