import torch
import argparse
from data_method import config_import, save_model, plot_accuracies, plot_losses, plot_lrs, predict

from torchvision.datasets import ImageFolder

from module import ImageFolderSubset, DeviceDataLoader, to_device, reload_dataset, evaluate, get_lr, Network, load_dataset
from torchvision.transforms import transforms
from torchsummary import summary

from tqdm import tqdm
import os
import torch.nn as nn

def train(train_dl, valid_dl):
    torch.cuda.empty_cache()
    history = []
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, epochs=epochs, steps_per_epoch=len(train_dl))
    print('train_dl len: ' + str(len(train_dl)))
    
    for epoch in tqdm(range(epochs)):
        
        model.train()
        # check for every 5 epochs
        if epoch % 5 == 0:
            save_model(args, model, optimizer, epoch, model_path)
            # reload the dataset
            train_dl, valid_dl = load_dataset(dataset, n_train, n_valid, batch_size, num_workers, device)
            
        
        train_losses = []
        lrs = []
        
        for batch in train_dl:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            
            if args.grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            # recording and updating learning rates
            lrs.append(get_lr(optimizer))
            sched.step()
            
        # validation
        result = evaluate(model, valid_dl)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.training_epoch_end(epoch, result)
        history.append(result)
    
    save_model(args, model, optimizer, start_epoch + epoch, model_path)  
    return history
    
if __name__ =='__main__':
    
    # read parameters from config.yaml
    parser = argparse.ArgumentParser()
    config = config_import("config/config0.1.yaml")
    for option, content in config.items():
        parser.add_argument(f"--{option}", default=content, type=type(content))
    args = parser.parse_args()
    
    train_dir = args.train_dir
    test_dir = args.test_dir
    ratio = float(args.ratio)
    picture_size = args.picture_size
    
    batch_size = args.batch_size
    num_workers = args.num_workers
    start_epoch = args.start_epoch
    epochs = args.epochs
    model_name = args.model_name
    model_path = args.model_path + model_name
    resnet_type = args.resnet_type
    
    weight_decay = float(args.weight_decay)
    learning_rate = float(args.learning_rate)
    
    plt_path = args.plt_path + model_name
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(plt_path):
        os.makedirs(plt_path)
    
    transform = transforms.Compose([
            transforms.RandomResizedCrop(picture_size), 
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = ImageFolder(train_dir, transform=transform)
    n_data = len(dataset)
    n_train = int(n_data*ratio)
    n_valid = n_data - n_train
    class_num = len(dataset.classes)
    print("data number:" + str(n_data))
    print("class_num:" + str(class_num))
    
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dl, valid_dl = load_dataset(dataset, n_train, n_valid, batch_size, num_workers, device)
    
    model = to_device(Network(resnet_type, class_num), device) 
    print(model)
    
    INPUT_SHAPE = (3, 224, 224)
    print(summary(model.cuda(), (INPUT_SHAPE)))
    
    optimizer = torch.optim.Adam(model.parameters(), 0.01, weight_decay=weight_decay)
    
    if args.reload:
        model_check = os.path.join(args.model_path, "checkpoint_{}.tar".format(start_epoch))
        checkpoint = torch.load(model_check)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print("Reload from epoch " + start_epoch)
    loss_device = torch.device("cuda")
    
    # one step before training
    history = [evaluate(model, valid_dl)]
    print("Initial Result: ")
    print(history)
    
    # start training
    history += train(train_dl, valid_dl)
    
    test = ImageFolder(test_dir, transform=transform)
    
    predict(test_dir, test, model, dataset, model_name, device)
    
    plot_accuracies(history, plt_path)
    plot_lrs(history, plt_path)
    plot_losses(history, plt_path)
    
    