from argparse import ArgumentParser

import torch
import yaml
from matplotlib import pyplot as plt


def save_model(model, path, history=None):
    """
    Saves given model to specified path with or without learning history
    """

    torch.save({
        'model_state_dict': model.state_dict(),
        'learning_history': history}, path)


def load_model(model, path, device):
    """
    Loads parameters from trained model from specified path and
    updates parameters of a given model.
    """

    trained_model = torch.load(path, map_location=device)
    model.load_state_dict(trained_model['model_state_dict'])
    history = trained_model['learning_history']

    return history


def plot_learning_history(history, title=''):
    learning_curve, acc_train_curve, acc_val_curve = history
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    fig.suptitle(title)

    axes[0].plot(learning_curve)
    axes[0].set_title('Learning Curve')

    axes[1].plot(acc_train_curve, label='Train')
    axes[1].plot(acc_val_curve, label='Val')
    axes[1].legend()
    axes[1].set_title('Max accuracy on val set: {:.4f}'.format(max(acc_val_curve)))

def get_config(path):

    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def parse_args():

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', help='Path to configuration file', type=str, default=None)
    arg_parser.add_argument('--paths', help='Path to configuration file with paths', type=str, default=None)
    arg_parser.add_argument('--fold', help='Index of the validation fold', type=int, default=None)

    return arg_parser.parse_args()


class Logger(object):
    """
    Instance of this class sums values of e.g. the loss function over
    different iterations in one epoch, compute the average value of loss for
    this epoch and save it to Python list.
    """

    def __init__(self):
        self.last = 0.0
        self.average = 0.0
        self.sum = 0.0
        self.count = 0
        self.history = []

    def update(self, value):
        """
        Update the state of the logger instance:
        - add value to the self.sum attribute
        - increment count of seen values
        - reestimate average value

        This should must be called after each iteration over a mini-batch.

        value (int or float): value to be logged
        """

        self.count += 1
        self.last = value
        self.sum += value
        self.average = self.sum / self.count

    def reset(self):
        """
        Zero-out all attributes except self.history.

        This method should be called at the begining of each epoch.
        """

        self.last = 0.0
        self.average = 0.0
        self.sum = 0.0
        self.count = 0

    def save(self):
        """
        Save the obtained average value to the list.

        This method should be called at the end of each epoch.
        """

        self.history.append(self.average)