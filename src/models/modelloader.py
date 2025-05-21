
import inspect
import importlib

from src.models import *
from .resnet import fullrank_resnet_loader
from .lenet import LeNet

def load_model(args):
    # get model instance
    if 'ResNet' in args.model_name:
        model = fullrank_resnet_loader(args)
    elif 'LeNet' in args.model_name:
        model = LeNet(args)

    return model, args
