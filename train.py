import os
import argparse
import torch
import options
import utils
import time
import numpy as np
from model import SimpleCNN
from utils import Bar, label2index, ECGDataloader, Writer, save_checkpoint
from torch.utils.data import DataLoader
import random
from sklearn.metrics import f1_score

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.model = SimpleCNN(opt).to(opt.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr_initial)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.decay_epoch, gamma=0.1)
        self.writer = Writer(self._get_tboard_dir())
        self.train_loader, self.valid_loader = self._load_data()
        self.log_file_path = os.path.join(self._get_tboard_dir(), 'training_log.txt')

        # Load pretrained model if specified
        if self.opt.pretrained:
            self._load_pretrained_model()

    # Z-score normalization
    def _normalize_ecg(self, ecg_data, axis=1):
        mean = np.mean(ecg_data, axis=axis, keepdims=True)
        std = np.std(ecg_data, axis=axis, keepdims=True)
        return (ecg_data - mean) / (std + 1e-8)  # Prevent division by zero

    def _load_data(self):
        # Load and preprocess data
        train_data = np.load(self.opt.path_train_data)  # Load ECG train data
        train_labels = np.load(self.opt.path_train_labels)  # Load train labels

        val_data = np.load(self.opt.path_val_data)  # Load ECG validation data
        val_labels = np.load(self.opt.path_val_labels)  # Load validation labels

        Y_train = np.array([label2index(i) for i in train_labels])  # Convert labels to indices
        Y_val = np.array([label2index(i) for i in val_labels])  # Convert labels to indices

        # Normalize data along the time axis
        train_data = self._normalize_ecg(train_data)
        val_data = self._normalize_ecg(val_data)

        # Expand dimensions to match model input requirements
        X_train, X_val = np.expand_dims(train_data, 1), np.expand_dims(val_data, 1)

        # Create DataLoader for training and validation
        train_loader = DataLoader(ECGDataloader(X_train, Y_train), batch_size=self.opt.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(ECGDataloader(X_val, Y_val), batch_size=self.opt.batch_size, shuffle=False,
                                  num_workers=0)

        return train_loader, valid_loader

    def _get_tboard_dir(self):
        # Initialize directories for logging and model storage
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log', f'{self.opt.log_name}')
        utils.mkdir(log_dir)
        utils.mkdir(os.path.join(log_dir, 'logs'))
        utils.mkdir(os.path.join(log_dir, 'models'))
        return os.path.join(log_dir, 'logs')

    def _load_pretrained_model(self):
        # Load pretrained model weights if specified
        print('Loading the pretrained model...')
        chkpt = torch.load(self.opt.pretrained_model)
        self.model.load_state_dict(chkpt['model'])
        self.optimizer.load_state_dict(chkpt['optimizer'])
        utils.optimizer_to(self.optimizer, self.opt.device)
        print('Resuming Start Epoch:', chkpt['epoch'] + 1)

    def train(self):
        # Print the total number of parameters in the model
        print(
            f'Total parameters: {utils.cal_total_params(self.model):,} ({utils.cal_total_params(self.model) / 1e6:.2f}M)')
        best_f1 = 0
        for epoch in range(1, self.opt.nepoch + 1):
            start_time = time.time()
            self.model.train()
            train_loss = 0

            # Training loop
            for X, Y in Bar(self.train_loader):
                X, Y = X.float().to(self.opt.device), Y.long().to(self.opt.device)  # Move data to device

                # Forward pass and optimization
                outputs = self.model(X)  # Get model predictions
                loss = self.loss_fn(outputs, Y)  # Calculate loss
                self.optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Backpropagate to calculate gradients
                self.optimizer.step()  # Update model parameters
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)  # Calculate average training loss
            self.writer.log_train_loss('total', avg_train_loss, epoch)  # Log training loss

            # Validation
            accuracy, f1, avg_val_loss = self._evaluate(self.valid_loader, epoch)  # Evaluate model on validation set
            if f1 > best_f1:  # Save the best model based on accuracy
                best_f1 = f1
                save_checkpoint(self._get_model_dir(), self.model, epoch)

            self.writer.log_score('F1-score', f1, epoch)  # Log validation accuracy
            self.scheduler.step()  # Update learning rate scheduler

            # Logging
            log_message = (
                f'EPOCH[{epoch}] Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f} | Validation F1-score: {f1:.6f} | Time: {time.time() - start_time:.3f}s'
            )
            print(log_message)
            self._log_to_file(log_message)

        print('Training completed.')

    def _evaluate(self, dataloader, epoch):
        # Evaluate the model on the given dataloader
        self.model.eval()
        pred_labels, true_labels = [], []
        total_loss = 0
        with torch.no_grad():
            for X, Y in Bar(dataloader):
                X, Y = X.float().to(self.opt.device), Y.long().to(self.opt.device)  # Move data to device
                pred = self.model(X)  # Get model predictions
                loss = self.loss_fn(pred, Y)  # Calculate loss
                total_loss += loss.item()

                # Get predicted class directly from raw logits
                pred_classes = torch.argmax(pred, dim=1)
                pred_labels.extend(pred_classes.cpu().numpy())
                true_labels.extend(Y.cpu().numpy())

        # Calculate accuracy
        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)
        accuracy = np.mean(pred_labels == true_labels)  # Calculate accuracy
        avg_valid_loss = total_loss / len(dataloader)  # Calculate average validation loss
        f1 = f1_score(true_labels, pred_labels, average='macro')  # Calculate F1-score
        self.writer.log_valid_loss('total', avg_valid_loss, epoch)  # Log validation loss
        return accuracy, f1, avg_valid_loss

    def _get_model_dir(self):
        # Get directory path for saving models
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log',
                               f'{self.opt.log_name}')
        return os.path.join(log_dir, 'models')

    def _log_to_file(self, message):
        # Write log message to file
        with open(self.log_file_path, 'a') as f:
            f.write(message + '\n')


if __name__ == '__main__':
    # Parse command-line arguments
    opt = options.Options().init(argparse.ArgumentParser(description='ECG Classification')).parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    # Initialize trainer and start training
    trainer = Trainer(opt)
    trainer.train()

