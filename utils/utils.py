from transformers import  TrOCRProcessor, VisionEncoderDecoderModel
import torch

from time import gmtime, strftime
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Logger:
    def __init__(self, state, max_epoch, dataloader_len, gpu=0):
        self.state = state  # "Train" / "Val"
        self.max_epoch = max_epoch
        self.dataloader_len = dataloader_len
        self.gpu = gpu
       

    def log_iteration(self, epoch, iter_idx, loss=None):
        if self.gpu != 0:
            return

        if loss is not None:
            self.update_loss(loss)

        log_str = f"[{strftime('%Y-%m-%d %H:%M:%S', gmtime())}] "
        log_str += f"{self.state} Epoch [{epoch}/{self.max_epoch}] "
        log_str += f"Iter [{iter_idx}/{self.dataloader_len}]"

        if loss is not None:
            log_str += f", Loss: {loss}"

        print(log_str)

    def log_metrics(self, epoch, cer=None):
        if self.gpu != 0:
            return

        log_str = f"[{strftime('%Y-%m-%d %H:%M:%S', gmtime())}] "
        log_str += f"{self.state} Epoch [{epoch}/{self.max_epoch}]"

        if cer is not None:
            log_str += f", CER: {cer:.6f}"

        print(log_str)

class write_logg:
    def __init__(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        self.file = open(file_path, "a")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


def modeluse(name_model, cfg):

    if(name_model == "trocr"):
        model = VisionEncoderDecoderModel.from_pretrained(cfg.model).to(device)
        process = TrOCRProcessor.from_pretrained(cfg.processor)
    else:
        print("don't have name model")
    return model, process