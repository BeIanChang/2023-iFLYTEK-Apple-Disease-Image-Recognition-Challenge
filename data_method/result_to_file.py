import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from module import predict_image

def plot_accuracies(history, address):
    accuracies = [x['val_accuracy'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig(address + '/' + 'acc.png', dpi=300)

def plot_losses(history, address):
    train_losses = [x.get('train_loss').cpu().numpy() for x in history]
    val_losses = [x['val_loss'].cpu().numpy() for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(address + '/' +'loss.png', dpi=300)
    
def plot_lrs(history, address):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
    plt.savefig(address + '/' + 'lr.png', dpi=300)
    
def predict(test_dir, test, model, dataset, model_name, device):
    ids = sorted(os.listdir(test_dir + '/test'))
    uuids = []
    labels = []

    for i, (image, label) in enumerate(test):
        predict_label = predict_image(image, model, dataset, device)
        uuids.append(ids[i])
        labels.append(predict_label)
        
    # 将预测值和真实值写入csv文件
    with open('results_{}.csv'.format(model_name), 'w', newline='') as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(['uuid', 'label'])

        # 写入每个测试样本的标记和预测值
        for uuid, label in zip(uuids, labels):
            writer.writerow([uuid, label])