import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


import torch

from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(100)

import warnings
warnings.filterwarnings("ignore")

#**********************************************************************

df = pd.read_csv("/kaggle/input/squad-csv-format/SQuAD_csv.csv")
df.columns

df = df[['context','question', 'text']]

print("Number of records: ", df.shape[0])


df["context"] = df["context"].str.lower()
df["question"] = df["question"].str.lower()
df["text"] = df["text"].str.lower()

df.head()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
INPUT_MAX_LEN = 512 # Input length
OUT_MAX_LEN = 128 # Output Length
TRAIN_BATCH_SIZE = 8 # Training Batch Size
VALID_BATCH_SIZE = 2 # Validation Batch Size
EPOCHS = 5 # Number of Iteration

MODEL_NAME = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length= INPUT_MAX_LEN)

print("eos_token: {} and id: {}".format(tokenizer.eos_token, tokenizer.eos_token_id)) # End of token (eos_token)
print("unk_token: {} and id: {}".format(tokenizer.unk_token, tokenizer.eos_token_id)) # Unknown token (unk_token)
print("pad_token: {} and id: {}".format(tokenizer.pad_token, tokenizer.eos_token_id)) # Pad token (pad_token)


class T5Dataset:

    def init(self, context, question, target):
        self.context = context
        self.question = question
        self.target = target
        self.tokenizer = tokenizer
        self.input_max_len = INPUT_MAX_LEN
        self.out_max_len = OUT_MAX_LEN

    def len(self):
        return len(self.context)

    def getitem(self, item):
        context = str(self.context[item])
        context = " ".join(context.split())

        question = str(self.question[item])
        question = " ".join(question.split())

        target = str(self.target[item])
        target = " ".join(target.split())
        
        
        inputs_encoding = self.tokenizer(
            context,
            question,
            add_special_tokens=True,
            max_length=self.input_max_len,
            padding = 'max_length',
            truncation='only_first',
            return_attention_mask=True,
            return_tensors="pt"
        )
        

        output_encoding = self.tokenizer(
            target,
            None,
            add_special_tokens=True,
            max_length=self.out_max_len,
            padding = 'max_length',
            truncation= True,
            return_attention_mask=True,
            return_tensors="pt"
        )


        inputs_ids = inputs_encoding["input_ids"].flatten()
        attention_mask = inputs_encoding["attention_mask"].flatten()
        labels = output_encoding["input_ids"]

        labels[labels == 0] = -100  # As per T5 Documentation

        labels = labels.flatten()

        out = {
            "context": context,
            "question": question,
            "answer": target,
            "inputs_ids": inputs_ids,
            "attention_mask": attention_mask,
            "targets": labels
        }


        return out    
    

class T5DatasetModule(pl.LightningDataModule):

    def init(self, df_train, df_valid):
        super().init()
        self.df_train = df_train
        self.df_valid = df_valid
        self.tokenizer = tokenizer
        self.input_max_len = INPUT_MAX_LEN
        self.out_max_len = OUT_MAX_LEN


    def setup(self, stage=None):

        self.train_dataset = T5Dataset(
        context=self.df_train.context.values,
        question=self.df_train.question.values,
        target=self.df_train.text.values
        )

        self.valid_dataset = T5Dataset(
        context=self.df_valid.context.values,
        question=self.df_valid.question.values,
        target=self.df_valid.text.values
        )
      
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
         self.train_dataset,
         batch_size= TRAIN_BATCH_SIZE,
         shuffle=True, 
         num_workers=4
        )


    def val_dataloader(self):
        return torch.utils.data.DataLoader(
         self.valid_dataset,
         batch_size= VALID_BATCH_SIZE,
         num_workers=1
        )    
    

    
class T5Model(pl.LightningModule):
    
    def init(self):
        super().init()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):

        output = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

        return output.loss, output.logits


    def training_step(self, batch, batch_idx):

        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        labels= batch["targets"]
        loss, outputs = self(input_ids, attention_mask, labels)

        
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        labels= batch["targets"]
        loss, outputs = self(input_ids, attention_mask, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return loss


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)
 

def run():
    
    df_train, df_valid = train_test_split(
        df[0:10000], test_size=0.2, random_state=101
    )
    
    df_train = df_train.fillna("none")
    df_valid = df_valid.fillna("none")
    
    df_train['context'] = df_train['context'].apply(lambda x: " ".join(x.split()))
    df_valid['context'] = df_valid['context'].apply(lambda x: " ".join(x.split()))
    
    df_train['text'] = df_train['text'].apply(lambda x: " ".join(x.split()))
    df_valid['text'] = df_valid['text'].apply(lambda x: " ".join(x.split()))
    
    df_train['question'] = df_train['question'].apply(lambda x: " ".join(x.split()))
    df_valid['question'] = df_valid['question'].apply(lambda x: " ".join(x.split()))

   
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    dataModule = T5DatasetModule(df_train, df_valid)
    dataModule.setup()

    device = DEVICE
    models = T5Model()
    models.to(device)

    checkpoint_callback  = ModelCheckpoint(
        dirpath="/kaggle/working",
        filename="best_checkpoint",
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        callbacks = checkpoint_callback,
        max_epochs= EPOCHS,
        gpus=1,
        accelerator="gpu"
    )

    trainer.fit(models, dataModule)

run()


train_model = T5Model.load_from_checkpoint("/kaggle/working/best_checkpoint-v1.ckpt")

train_model.freeze()

def generate_question(context, question):

    inputs_encoding =  tokenizer(
        context,
        question,
        add_special_tokens=True,
        max_length= INPUT_MAX_LEN,
        padding = 'max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_tensors="pt"
        )

    
    generate_ids = train_model.model.generate(
        input_ids = inputs_encoding["input_ids"],
        attention_mask = inputs_encoding["attention_mask"],
        max_length = INPUT_MAX_LEN,
        num_beams = 4,
        num_return_sequences = 1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        )

    preds = [
        tokenizer.decode(gen_id,
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True)
        for gen_id in generate_ids
    ]

    return "".join(preds)
context = "pixels student activity started in school of engineering helwan university 11 years ago ,the role of this student activity is to fill the gap between the acadmic knowldge of the college and the industy experiance , so, we are the acadmic projects team in the student activity pixels we made this chatbot who are answering you now , which have been created by the brilliant genius Ahmed Hassan"

que = " how created this chatbot?"

print(generate_question(context, que))
