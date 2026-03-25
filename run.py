import torch

from typing import Optional, Tuple
import argparse

from time import gmtime, strftime

from utils.utils import Logger, write_logg, modeluse
# from utils.ddp_utils import ddp_setup
from utils.train_utils import Training, dataloader
from utils.train_transformers import Transformer_trainning, sel_attributes, select_dataloader

from data.iamdataset import process
from config.cnn_rnn import Confg
from config.trocr_fineturn import CONFIG_trocr
from net.model import CNN_RNN


def parse_arguments(params: Optional[Tuple] = None):
    parse =argparse.ArgumentParser(description = " Model deeplearing")
    parse.add_argument("-c", "--command", required= True, type = str, help = "Training or testing pipeline", choices= ['train', "test"])
    parse.add_argument("-n", "--model_name", required= True, type = str, help = "selec model for training", choices= ['CNN_RNN', "trocr"])
    # parse.add_argument('-p', "--path-to-config", required=True, type = str, help = "Path to config")
    parse.add_argument("--gpu", required=False, type = int, default= 1, help = "GPU to use")
    know_args , _ = parse.parse_known_args(params)
    return know_args


def run(args):

    

  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir ="dataset/iam_words/words"
    word_path = "dataset/iam_words/words_new.txt"
    if args.model_name == "CNN_RNN":
        config =Confg()
        model = CNN_RNN(config.cnn_config, config.head_cfg,len(config.classes_)).to(device)
        dataset = process(root_dir, word_path,config, args.model_name)
        train_loader, val_loader = dataloader(dataset, config)
   
        log_file = write_logg(f"logs/{strftime('%Y_%m_%d_%H_%M_%S', gmtime())}train.log")
        trainer = Training(
            model,
            config,
            train_loader,
            val_loader,
        
        )
        if args.command == "train":
            trainer.train()
        # if args.command == "test":
        #     trainer.test()

    else:
        if args.model_name == "trocr":
            config = CONFIG_trocr
        model, processor = modeluse(args.model_name, config)

        dataset = process(root_dir, word_path, config, args.model_name)
        train_loader, val_loader = select_dataloader(dataset, config)
        
        sel_attributes(model, processor, config)
        
        # log_file = write_logg(f"logs/{strftime('%Y_%m_%d_%H_%M_%S', gmtime())}train.log")
        # kaggel
        log_file = write_logg(f"/kaggle/working/logs/{strftime('%Y_%m_%d_%H_%M_%S', gmtime())}train.log")
        
        # self, model, processor, train_dataloader, val_dataloader, cnfg, device)
        trainer = Transformer_trainning(
            model, processor, train_loader, val_loader, config,device
        )

        if args.command == "train":
            trainer.train()
        # if args.command == "test":
        #     trainer.test()

    log_file.close()
if __name__ == "__main__":
    args = parse_arguments()
    run(args)