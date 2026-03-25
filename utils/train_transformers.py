import torch 
from torch.utils.data import random_split, DataLoader
from datasets import load_metric
from transformers import AdamW

from utils.utils import Logger, modeluse
from data.iamdataset import process
from tqdm  import tqdm
import time
import os
# import wandb



cer_metric = load_metric("cer")

def select_dataloader(dataset, cfg):
    train_size = cfg.train_size
    val_size = cfg.val_size

    train_set, val_set = random_split(dataset, [int(len(dataset) * train_size)+1, int(len(dataset) *val_size)])
    train_dataloder = DataLoader(train_set, batch_size = 4, shuffle = True)
    val_dataloder = DataLoader(val_set, batch_size = 4)
    return train_dataloder, val_dataloder

def sel_attributes(model, processor, cnfg):
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = cnfg.max_length
    model.config.early_stopping = cnfg.early_stopping
    model.config.no_repeat_ngram_size = cnfg.no_repeat
    model.config.length_penalty = cnfg.length_penalty
    model.config.num_beams = cnfg.num_beams



def computer_cer(pred_ids, label_ids, processor):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer
def Validation(i,model, val_dataloader, processor, device):
    model.eval()
    val_cer = 0.0
    with torch.no_grad():
        for  batch in tqdm(val_dataloader):
            output = model.generate[batch["pixel_val"]].to(device)
            cer = computer_cer(output, batch["labels"], processor)
            val_cer += cer
    
    cur_cer = val_cer/ len(val_dataloader)
    return cur_cer
    



class Transformer_trainning:
    def __init__(self, model, processor, train_dataloader, val_dataloader, cnfg, device):
        self.cfg = cnfg
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr = self.cfg.lr)
    

    def train(self):

        self.model.train()
        max_epochs = self.cfg.max_epochs
        logg = Logger("Train",max_epochs, len(self.train_dataloader), self.device )
        for epoch in range(max_epochs):
            start_time = time.time()
            train_loss = 0.0
            test_loss = 0.0
            
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                
                output = self.model(**batch)
                loss = output.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.optimizer.step()
                train_loss += loss.item() 

                if (i+1)% self.cfg.display_step ==0:
                    logg.log_iteration(epoch + 1, (i+1) * self.cfg.display_step, loss.item())
            print(f"Done epoch #{epoch+1}, time for this epoch: {time.time()-start_time}s")
            train_loss/= (i + 1)
            

            cur_cer = Validation(i,self.model, self.val_dataloader, self.processor, self.device)
            logg.log_metrics(epoch, cur_cer)
            os.makedirs(os.path.dirname(self.cfg.save_model), exist_ok= True)
            test_loss/= (i+1)
            print(f"Train Loss: {train_loss}, val_loss: {test_loss}")
            save_path = f"{self.cfg.save_model}/model_epoch_{epoch+1}.pth"
            torch.save(self.model.state_dict(), save_path)
        final_path = f"{self.cfg.save_model}/model_final.pth"
        torch.save(self.model.state_dict(), final_path)
        print(f"Final model saved at {final_path}")
    def test(self):
        pass

