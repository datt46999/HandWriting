class CONFIG_donut:
    model = "naver-clova-ix/donut-base"
    tokenizer = "xlm-roberta-base"
    height = 384
    width = 384
    # select dataset
    max_target =128
    train_size = 0.8
    val_size = 0.2


    # attribtes
    max_length = 64
    early_stopping = True
    no_repeat = 3 
    length_penalty =2.0
    num_beams = 4
    

    # training
    max_epochs = 5
    lr = 1e-5
    display_step = 1000
    
    save_model= "word_dir/"
    # save_model= "/kaggle/working/word_dir/"
