import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sweeps_result
import os
import torch
from datasets import Dataset
import json
from IPython.display import display
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification,TrainerCallback
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import torch.nn.functional as F
from utils import *
import io
import matplotlib.pyplot as plt
import params

class WandbMetricsLogger(TrainerCallback):
    def on_evaluate(self, args, state, control, model, metrics):
        # Log metrics to Wandb
        wandb.log(metrics)
        
default_config = {
        'method': 'random',
        'metric': {
        'goal': 'minimize', 
        'name': 'eval_loss'
        },
    }


    # hyperparameters
parameters_dict = {
        'epochs': {
            'value': 2
            },
        'seed': {
            'value': 42
            },
        'batch_size': {
            'values': [4, 8, 16]
            },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 2e-3
        },
        'weight_decay': {
            'values': [0.0, 0.2]
        },
        'learning_sch': {
            'values': ['linear','polynomial','cosine']
        },
        'architecture': {
            'values': ["distilroberta-base","bert-base-uncased","distilbert-base-uncased"]
        },
    }


default_config['parameters'] = parameters_dict

def compute_metrics_fn(eval_preds):
    metrics = dict()

    # Extract the validation loss from eval_preds
    validation_loss = eval_preds.loss
    metrics['validation_loss'] = validation_loss

    return metrics

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--batch_size', type=int, default=default_config.get("parameters").get("batch_size").get("values")[-1],
                           help='batch size')
    argparser.add_argument('--epochs', type=int, default=default_config.get("parameters").get("epochs").get("value"),
                           help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.get("parameters").get("learning_rate").get("min"),
                           help='learning rate')
    argparser.add_argument('--seed', type=int, default=default_config.get("parameters").get("seed").get("value"),
                           help='random seed')
    argparser.add_argument('--weight_decay', type=float, default=default_config.get("parameters").get("weight_decay").get("values")[-1],
                           help='random seed')
    
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return



def train(config=None):
    
    torch.manual_seed(default_config.get("parameters").get("seed").get("value"))
    
    run = wandb.init(
                project=params.WANDB_PROJECT, 
                entity=params.ENTITY, 
                   job_type="train",
                name = "04-Retrain",
                tags = ["RETRAIN"]
                
    )
    if "artifacts" not in os.listdir():
        raw_data_at = run.use_artifact(params.RAW_DATA_AT, 
                                                       type='raw_data')
        artifact_di = raw_data_at.download()
    else: pass
    train , test = read_data()
    dataset_train, dataset_test = preprocess(train=train,test=test)
    if config is None:
        config = wandb.config
    else:
        pass 
    split_train_dataset,split_eval_dataset , tokenized_test , tokenizer, df_val = dataset_tokenize_n_split(train,dataset_train, dataset_test,config.architecture)

    
    
    
    model = create_model(model_name =config.architecture ,num_labels = 7)
    
    
    training_args = TrainingArguments(                                

                                output_dir='distilroberta-retrain',
                                report_to='wandb',  # Turn on Weights & Biases logging
                                num_train_epochs=config.epochs,
                                learning_rate=config.learning_rate,
                                lr_scheduler_type = config.learning_sch,
                                metric_for_best_model="eval_loss", 
                                load_best_model_at_end=True,
                                remove_unused_columns=True,
                                greater_is_better=False,
                                weight_decay = config.weight_decay,
                                evaluation_strategy="steps",
                                logging_steps=100,
                                per_device_train_batch_size = config.batch_size,
                                per_device_eval_batch_size = config.batch_size ,
                                
                                )
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=split_train_dataset,
                        eval_dataset=split_eval_dataset,
                        callbacks = [early_stopping],
                        tokenizer=tokenizer,
        )
    trainer.train()
    
    val_pred = predict_fn(model, split_eval_dataset)
    conf_mat(df_val,val_pred)

    wandb.finish()
    train(sweeps_result)

if __name__ == '__main__':
    parse_args()
    train(default_config)