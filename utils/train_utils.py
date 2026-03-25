import torch
import editdistance
from torch.utils.data import random_split, DataLoader
from torch import nn, optim



import tqdm
from net.model import CNN_RNN
from data.iamdataset import process
from utils.utils import Logger
import os
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)

    return images, labels, label_lengths

def dataloader(dataset, cfg):
    train_size = cfg.train_size
    train_len = int(len(dataset) * train_size)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size = 4, collate_fn = collate_fn, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = 4, collate_fn = collate_fn, shuffle = False)
    return train_loader, val_loader


def Eval(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    eval_cer =0.0
    total_character_count = 0.0
    total_edit_distance =0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(val_loader)):
            images, labels, label_lengths= batch
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            output,_ = model(images)
            log_probs = output.log_softmax(2)

            T, N, C = log_probs.size()
            
            preds = torch.argmax(log_probs, dim =2)
            preds = preds.permute(1,0)

            labels_cpu = labels.cpu().numpy()
            label_lengths_cpu = label_lengths.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            idx = 0

           
            # compute CER
            for n in range(N):
            
                l = label_lengths_cpu[n]
                target = labels_cpu[idx : idx + l].tolist()
                idx += l

            
                pred_raw = preds_cpu[n]
                decoded = []
                prev = -1
                for p in pred_raw:
                    if p != 0 and p != prev: 
                        decoded.append(p)
                    prev = p

                distance = editdistance.eval(decoded, target)
                
                total_edit_distance += distance
                total_character_count += len(target)

        
        print(total_character_count)
        print(total_edit_distance)
        print(f"Sample Target: {target}")
        print(f"Sample Decoded: {decoded}")
        print(f"Total Character Count: {total_character_count}")
        return total_edit_distance/ total_character_count


class Training:
    def __init__(self,
                 model : CNN_RNN,
                 cfg, 
                 train_loader, val_loader,          
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)

    def train(self):

        max_epoch = self.cfg.max_epoch
        logg = Logger("Train",max_epoch, len(self.train_loader), self.device )
        for epoch in  range(max_epoch):
            print(f"Training : {epoch+1}")
            self.model.train()
            total_loss_train = 0.0
            
            for i, batch in enumerate(tqdm.tqdm(self.train_loader)):

                images, labels, label_lengths = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                label_lengths = label_lengths.to(self.device)

                
                output,_ = self.model(images)
                log_probs =  output.log_softmax(2)
                T,N, C = log_probs.size()
                input_lengths = torch.full(
                    size=(N,),
                    fill_value=T,
                    dtype=torch.long
                ).to(self.device)

                # compute loss
                loss = self.criterion(
                    log_probs,
                    labels,
                    input_lengths,
                    label_lengths
                )

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss_train += loss.item()
                
                if (i + 1) % 100 == 0:
                    current_avg_loss = total_loss_train / (i + 1)
                    logg.log_iteration(epoch+1, i+1, current_avg_loss)
            avg_epoch_loss = total_loss_train / len(self.train_loader)
            cer = Eval(self.model, self.val_loader)
            
            logg.log_metrics(epoch+1, cer)
            os.makedirs(os.path.dirname(self.cfg.save_model), exist_ok= True)
            save_path = f"{self.cfg.save_model}/model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_epoch_loss
            }, save_path)
            print(f"Model saved at {save_path}")
            logg.log_iteration(epoch+1, len(self.train_loader), avg_epoch_loss)
            logg.log_metrics(epoch+1, cer)

        # --- Optionally save final model separately ---
        final_path = f"{self.cfg.save_model}/model_final.pth"
        torch.save(self.model.state_dict(), final_path)
        print(f"Final model saved at {final_path}")
    def test(self):
        pass