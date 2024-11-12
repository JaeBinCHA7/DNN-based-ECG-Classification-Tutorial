import argparse
import torch
import options
import numpy as np
from model import SimpleCNN
from utils import label2index, ECGDataloader, Bar
from torch.utils.data import DataLoader
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score

class Tester:
    def __init__(self, opt):
        self.opt = opt
        self.model = SimpleCNN(opt).to(opt.device)
        self._load_pretrained_model()
        self.test_loader = self._load_data()

    # Z-score normalization
    def _normalize_ecg(self, ecg_data, axis=1):
        mean = np.mean(ecg_data, axis=axis, keepdims=True)
        std = np.std(ecg_data, axis=axis, keepdims=True)
        return (ecg_data - mean) / (std + 1e-8)  # Prevent division by zero

    def _load_data(self):
        # Load and preprocess data
        test_data = np.load(self.opt.path_test_data)  # Load ECG train data
        test_labels = np.load(self.opt.path_test_labels)  # Load train labels

        test_data = self._normalize_ecg(test_data)
        Y_test = np.array([label2index(i) for i in test_labels])

        # Prepare DataLoader
        X_test = np.expand_dims(test_data, 1)
        test_loader = DataLoader(ECGDataloader(X_test, Y_test), batch_size=self.opt.batch_size, shuffle=False, num_workers=0)
        return test_loader

    def _load_pretrained_model(self):
        # Load the pretrained model for evaluation
        print('Loading the pretrained model...')
        chkpt = torch.load(self.opt.pretrained_model, map_location=self.opt.device)
        self.model.load_state_dict(chkpt['model'])

    def test(self):
        self.model.eval()
        pred_labels, true_labels = [], []
        total_loss = 0
        loss_fn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for X, Y in Bar(self.test_loader):
                X, Y = X.float().to(self.opt.device), Y.long().to(self.opt.device)
                outputs = self.model(X)
                loss = loss_fn(outputs, Y)
                total_loss += loss.item()

                # Get predicted classes
                pred_classes = torch.argmax(outputs, dim=1)
                pred_labels.extend(pred_classes.cpu().numpy())
                true_labels.extend(Y.cpu().numpy())

        # Calculate accuracy
        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)
        accuracy = np.mean(pred_labels == true_labels)
        avg_test_loss = total_loss / len(self.test_loader)
        f1 = f1_score(true_labels, pred_labels, average='macro')  # Calculate F1-score

        # Print results in a more professional format
        print(f'==================== Test Results ====================')
        # print(f'| Test Accuracy    : {accuracy * 100:.2f}%')
        print(f'| Test F1-score    : {f1 * 100:.2f}%')
        print(f'=======================================================')

if __name__ == '__main__':
    opt = options.Options().init(argparse.ArgumentParser(description='ECG Classification Testing')).parse_args()
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    tester = Tester(opt)
    tester.test()

