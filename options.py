import argparse

class Options:
    def __init__(self):
        pass

    def init(self, parser):
        # Global settings
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size for training and validation.')
        parser.add_argument('--nepoch', type=int, default=50,
                            help='Number of training epochs.')
        parser.add_argument('--lr_initial', type=float, default=1e-4,
                            help='Initial learning rate for the optimizer.')
        parser.add_argument('--decay_epoch', type=int, default=20,
                            help='Epoch at which to start decaying the learning rate.')

        # Device settings
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to use for training ("cuda" for GPU, "cpu" for CPU).')

        # Model settings
        parser.add_argument('--classes', type=int, default=5,
                            help='Number of output classes for classification.')

        # Pretrained model settings
        parser.add_argument('--log_name', type=str, default='241111',
                            help='Identifier for logging and checkpointing.')
        parser.add_argument('--pretrained', type=bool, default=False,
                            help='Whether to load a pretrained model (True/False).')
        parser.add_argument('--pretrained_model', type=str,
                            default='./log/241111/models/ckpt_opt.pt',
                            help='Path to the pretrained model weights file.')

        # Dataset settings
        parser.add_argument('--fs', type=int, default=360,
                            help='Sampling frequency of the ECG data.')
        parser.add_argument('--path_train_data', type=str,
                            default='./dataset/train_data.npy',
                            help='Path to save the training data.')
        parser.add_argument('--path_train_labels', type=str,
                            default='./dataset/train_labels.npy',
                            help='Path to save the training labels.')
        parser.add_argument('--path_val_data', type=str,
                            default='./dataset/val_data.npy',
                            help='Path to save the validation data.')
        parser.add_argument('--path_val_labels', type=str,
                            default='./dataset/val_labels.npy',
                            help='Path to save the validation labels.')
        parser.add_argument('--path_test_data', type=str,
                            default='./dataset/test_data.npy',
                            help='Path to save the test data.')
        parser.add_argument('--path_test_labels', type=str,
                            default='./dataset/test_labels.npy',
                            help='Path to save the test labels.')


        return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options for ECG Classification Training')
    opt = Options().init(parser).parse_args()
    print(opt)
