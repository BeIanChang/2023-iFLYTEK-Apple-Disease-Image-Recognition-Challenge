import os
import torch


def save_model(args, model, optimizer, current_epoch, model_path):
    out = os.path.join(model_path + '/', "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
    print("checkpoints of {} saved.".format(current_epoch))
