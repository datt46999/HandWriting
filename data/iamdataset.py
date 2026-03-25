import torch
import torchvision

from torchvision import transforms
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image


from utils.utils import modeluse


def select2df(data_path):
    
    df = pd.read_csv(data_path, header=None, sep=" ", on_bad_lines='warn', comment='#')
    

    df = df.rename(columns={0: "file_name", 8: "text"})
    
    df = df[["file_name", "text"]]
   
    df = df[~df['text'].isin(['at', ',', ')'])]
    df = df[df['file_name'] != 'a01-117-05-02']
    df['text'] = df['text'].astype(str)
    

    df = df[df['text'].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)
    return df

def process_image(img, img_cfg):
    transform = transforms.Compose([
        transforms.Resize(img_cfg.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean= img_cfg.img_mean, std = img_cfg.img_std)
    ])
    return transform(img)

class IAM_dataset(Dataset):
    def __init__(self ,cfg, root_dir , df, model_name,max_target = 128):
        self.root_dir = root_dir
        self.df = df
        self.max_target = max_target
        self.cfg = cfg 
        self.model_name =  model_name
        self_,self.processor = modeluse(self.model_name, self.cfg)


    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        file_name = self.df["file_name"][idx]
        text = self.df["text"][idx]

        path = file_name.split('-')
        file_name = file_name + '.png' 
        file_path = self.root_dir +'/' +  path[0] + '/' +path[0]+'-'+path[1]+ '/'+ file_name
        
        # 

        if self.model_name == "CNN_RNN":
            image = Image.open(file_path).convert('L')
            pixel_val = process_image(image, self.cfg)
            labels = torch.tensor(
                [self.cfg.cdict.get(c, 0) for c in text],  # 0 = blank
                dtype=torch.long
            )
            
        else:
            image = Image.open(file_path).convert("RGB")
            # _, processor = modeluse(self.model_name, self.cfg)
            if self.processor is None:
                print("check name of model")
            try:
                pixel_values = self.processor(image, return_tensors = "pt").pixel_values
            except:
                pixel_values = torch.zeros((3, 384, 384))
            
            # labels = self.processor.tokenizer(text = labels, padding = "max_length", truncation = True, max_leght = self.max_target).input_ids
            labels = self.processor.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_target
            ).input_ids
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
            labels = labels + [-100] * (self.max_target  - len(labels)) 
            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
            return encoding

        return pixel_values, labels




        
        
def process(root_dir, word_path, cfg, model_name):
    df = select2df(word_path)
    encoding = IAM_dataset(cfg, root_dir, df, model_name)

    return encoding 