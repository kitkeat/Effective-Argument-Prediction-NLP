from flask import Flask, request
import os
import pandas as pd
# import numpy as np
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, AutoModel
from datasets import Dataset
import math

from sklearn.preprocessing import LabelEncoder

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


data = {'discourse_type':[''],'discourse_text':['']}
data_path = pd.DataFrame(data)
test_path = pd.DataFrame(data)


attributes = ["Adequate" ,"Effective","Ineffective"]
distilbert_config={'name': 'distilbert',
                   'model_name':'/distilbert-base-uncased',
                    'newly_tuned_model_path' : '/20220820-043647.pth',
                    'wandb':False,
                    'param':{
                      'n_labels': 3,
                      'batch_size': 64,
                      'lr': 8e-4,#6e-5,
                      'warmup': 0, 
                      'weight_decay': 0.01,#Default is 0.01
                      'n_epochs': 5,#4,
                      'n_freeze' : 5,
                      'p_dropout':0,#0.2,#0.6,
                      'scheduler':False,
                      'precision':16, #Default is 32
                      'sample_mode':True,
                      'sample_size': 100,
                        'swa':False,
                        'swa_lrs':1e-2
                        
                  }
              }

seed_everything(91, workers=True)


# Freeze the hidden layer within the pretrained model
def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False
        
def get_freezed_parameters(module):
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
    return freezed_parameters


class _Dataset(Dataset):
    def __init__(self,data_path,test_path, tokenizer,label_encoder,attributes,config, max_token_len: int = 512, is_train=True,is_test=False):
        self.data_path = data_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.attributes = attributes
        self.max_token_len = max_token_len
        self.is_train = is_train
        self.is_test = is_test
        self.label_encoder = label_encoder
        self.config = config
        self._prepare_data()

    def _prepare_data(self):
        SEP = self.tokenizer.sep_token # different model uses different to text as seperator (e.g. [SEP], </s>)
        df = self.test_path
        df['text'] = df['discourse_type'] + SEP + df['discourse_text']
        df = df.loc[:,['text']]
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        item = self.df.iloc[index]
        text = str(item.text)
        tokens = self.tokenizer.encode_plus(text,
                                  add_special_tokens= True,
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=self.max_token_len,
                                  return_attention_mask = True)
        if self.is_test:
            return {'input_ids':tokens.input_ids.flatten(),'attention_mask': tokens.attention_mask.flatten()}
        else:
            attributes = item['labels'].split()
            self.label_encoder.fit(self.attributes)
            attributes = self.label_encoder.transform(attributes)
            attributes = torch.as_tensor(attributes)
            return {'input_ids':tokens.input_ids.flatten(),'attention_mask': tokens.attention_mask.flatten(), 'labels':attributes}


class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["labels"] = [sample["labels"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s.tolist() + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s.tolist() + (batch_max - len(s)) * [0] for s in output["attention_mask"]]

        else:
            output["input_ids"] = [torch.FloatTensor((batch_max - len(s)) * [self.tokenizer.pad_token_id].tolist()) + s.tolist() for s in output["input_ids"]]
            output["attention_mask"] = [torch.FloatTensor((batch_max - len(s)) * [0]) + s.tolist() for s in output["attention_mask"]]
            
        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["labels"] = torch.tensor(output["labels"], dtype=torch.long)
        return output

class _Data_Module(pl.LightningDataModule):

    def __init__(self, data_path, test_path,attributes,label_encoder,tokenizer,config, batch_size: int = 8, max_token_length: int = 512):
        super().__init__()
        self.data_path = data_path
        self.test_path = test_path
        self.attributes = attributes
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.config = config

    def setup(self, stage = None):
        if stage == 'predict':
            self.test_dataset = _Dataset(self.data_path, self.test_path, label_encoder = self.label_encoder, attributes=self.attributes, is_train=False,is_test=True, tokenizer=self.tokenizer,config = self.config)

    def predict_dataloader(self):
        collate_fn = Collate(self.tokenizer, 
                           isTrain=False)

        return DataLoader(self.test_dataset, 
                        batch_size = self.batch_size, 
                        num_workers=2, 
                        shuffle=False,
                        collate_fn = collate_fn)


class DistilBert_Text_Classifier(pl.LightningModule):
    
    def __init__(self, config: dict,data_module):
        super().__init__()
        self.config = config
        self.data_module=data_module
        self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
        freeze((self.pretrained_model).embeddings)
        freeze((self.pretrained_model).transformer.layer[:config['param']['n_freeze']])
        self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['param']['n_labels'])
        self.loss_func = nn.CrossEntropyLoss() # do not put SoftMax, just use CrossEntropyLoss
        
        self.dropout = nn.Dropout(config['param']['p_dropout'])

    # For inference        
    def forward(self, input_ids, attention_mask, labels = None):
        output = self.pretrained_model(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)  # mean of sequence length
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
            
        loss = 0
        if labels is not None:
            loss = self.loss_func(logits,labels)
        return loss, logits

    def predict_step(self, batch, batch_index):
        loss, logits = self(**batch)
        return logits
    
    def configure_optimizers(self):
        train_size = len(self.data_module.train_dataloader())
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['param']['lr'], weight_decay=self.config['param']['weight_decay'])
        if self.config['param']['scheduler']:
            total_steps = train_size/self.config['param']['batch_size']
            warmup_steps = math.floor(total_steps * self.config['param']['warmup'])
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            return[optimizer],[scheduler]
        else:
            return optimizer
        
def predict(_Text_Classifier,config,test_path):
    attributes = ["Adequate" ,"Effective","Ineffective"]
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
    le = LabelEncoder()

    # Initialize data module
    test_data_module = _Data_Module(data_path,
                                    test_path,
                                    attributes,
                                    le,
                                    tokenizer,
                                    batch_size=config['param']['batch_size'],
                                    config=config
                                   )
    test_data_module.setup()

    # Initialize Model
    model = _Text_Classifier(config,test_data_module)
    model.load_state_dict(torch.load(config['newly_tuned_model_path']))

    # Initialize Trainer
    trainer = pl.Trainer(accelerator='auto')

    output = trainer.predict(model, datamodule=test_data_module)
    predictions = output[0].argmax(dim=-1).item()
    return predictions

@app.route('/')
def home():
    return {"success":True}, 200

# discourse_text = "easy plagiarism"

@app.route('/predict', methods = ['POST'])
def make_prediction():
    user_input = request.get_json(force=True)
    discourse_type = "Claim"
    discourse_text = user_input
    test_path = pd.DataFrame({'discourse_type':[discourse_type],'discourse_text':[discourse_text]})
    
    prediction = predict(DistilBert_Text_Classifier,distilbert_config,test_path)
    # prediction = int(discourse_text)
    if prediction == 0:
        out = 'Adequate'
    elif prediction == 1:
        out = 'Effective'
    elif prediction == 2:
        out = 'Ineffective'
    return {'response':out}


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=int(os.environ.get("PORT", 8080)))
