## 📖 Introduction 
This project focuses on handwritten English text recognition using the IAM dataset at the word level. The script is designed to train and evaluate multiple models, including a CNN-RNN architecture and advanced transformer-based approaches such as TrOCR and Donut, which are fine-tuned for improved performance on handwritten text. Final model was valudation with CER (Character Error Rate)

## 🛠️ Installation
### Dataset:
 **IAM Handwriting Database**: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database. 
    
    > This cannot be reachable, so we use the processed dataset from **Kaggle**: https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database
### Remember:
install torch, torchvision suitable for your computer


```shell
git clone git@github.com:datt46999/HandWriting.git
cd HandWriting
pip install -r requirements.txt
```
## 👨‍🏫 Get Started
Script include three model: CNN_RNN, trocr, donut
## Train
```shell
python run.py -c train - n <name_of_model>
```

## 👀 Model pretrained :
Since our model is too large, so you need to download manually.
Link model Pre-trained:





