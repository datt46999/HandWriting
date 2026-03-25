class Confg:
    # cnn_rnn
    classes_ = " '_ !#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    cdict = {c:i for i, c in enumerate(classes_)}
    icdict = {i:c for i, c in enumerate(classes_)}


    k = 1
    cnn_config =[(1, 64), "M", (2, 128), "M", (2, 256)]

    img_mean = [0.5]
    img_std = [0.5]


    head_type = "rnn"
    head_cfg = (256, 2)  #hidden , num_layers

    flattening = "maxpool"
    resize = (32,128)


    #  train
    train_size = 0.8
    
    display_step = 1000
    max_epoch = 5
    lr = 1e-5

    save_model = "word_dir/"
    # save_model = "kaggle/working/word_dir/"
    
    load_model = False


