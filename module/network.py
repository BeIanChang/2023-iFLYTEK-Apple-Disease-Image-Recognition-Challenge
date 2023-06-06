from .resnet import get_resnet
from torch import nn
import torch
import torch.nn.functional as F # for functions for calculating loss
from .evaluation import accuracy

class Network(nn.Module):
    def __init__(self, resnet_type, class_num):
        super(Network, self).__init__()
        self.resnet = get_resnet(resnet_type, class_num)
        self.class_num = class_num
        # pay attention, if not myResNet, rep_dim is needed. or classifier will receive 512
        if(resnet_type == 'myResNet'):
            self.rep_dim = 512
        else:
            self.rep_dim = self.resnet.rep_dim
        self.mlp = nn.Sequential(    # for otherwise myResNet
            nn.Linear(self.rep_dim, self.rep_dim), 
            nn.ReLU(),
            nn.Linear(self.rep_dim, self.class_num)
        )
        self.classifier = nn.Sequential(nn.MaxPool2d(4),  
                                       nn.Flatten(),
                                       nn.Linear(self.rep_dim, class_num),
                                       nn.Softmax(dim=1)
                                       )

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x
    
    def forward_max(self, x):
        h = self.resnet(x)
        c = self.mlp(h)
        c = torch.argmax(c, dim=1)
        return c
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    # output for every validating epoch end
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    
    # output for every training epoch end
    def training_epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))