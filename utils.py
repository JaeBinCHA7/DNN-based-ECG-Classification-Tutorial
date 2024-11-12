import os
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter


# For dataset
class ECGDataloader():  # 1110 - 4096 samples
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.float))

    def __len__(self):
        return len(self.data)


# For dataset
def label2index(i):
    m = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}  # uncomment for 5 classes
    return m[i]


# Create a new directory.
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Normalize the ECG data using Z-score normalization.
def normalize_ecg(ecg_data):
    mean = np.mean(ecg_data, axis=0, keepdims=True)
    std = np.std(ecg_data, axis=0, keepdims=True)
    return (ecg_data - mean) / (std + 1e-8)  # Prevent division by zero


# for using pre-training weights
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# Calculate total number of parameters in a model.
def cal_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


# Display a progress bar during training/validation.
class Bar(object):
    def __init__(self, dataloader):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloder.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloder.')

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 50

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())

        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)

        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()

        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._reset()

        return batch

    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0

        rate = self._idx / len(self.dataloader)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        bar = ('=' * len_bar + '>').ljust(self._DISPLAY_LENGTH, '.')
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')

        tmpl = '\r{}/{}: [{}] - ETA {:.1f}s'.format(
            idx,
            len(self.dataset),
            bar,
            eta
        )
        print(tmpl, end='')
        if self._batch_idx == len(self.dataset):
            print()

    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []


# Define a custom writer class that extends SummaryWriter to log training/validation metrics.
class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)

    # Method to log training loss.
    def log_train_loss(self, loss_type, train_loss, step):
        self.add_scalar('train_{}_loss'.format(loss_type), train_loss, step)

    # Method to log validation loss.
    def log_valid_loss(self, loss_type, valid_loss, step):
        self.add_scalar('valid_{}_loss'.format(loss_type), valid_loss, step)

    # Method to log other performance metrics (e.g., accuracy, F1-score).
    def log_score(self, metrics_name, metrics, step):
        # Add a scalar value to the writer with the given metric name.
        self.add_scalar(metrics_name, metrics, step)


def save_checkpoint(exp_log_dir, model, epoch):
    save_dict = {
        "model": model.state_dict(),
        'epoch': epoch
    }
    # save classification report
    save_path = os.path.join(exp_log_dir, "ckpt_opt.pt")

    torch.save(save_dict, save_path)
