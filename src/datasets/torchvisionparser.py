import torch
import torchvision
import logging


logger = logging.getLogger(__name__)


# dataset wrapper module
class VisionClassificationDataset(torch.utils.data.Dataset): 
    def __init__(self, dataset, dataset_name, suffix):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.suffix = suffix
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        return inputs, targets

    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return f'[{self.dataset_name}] {self.suffix}'

# helper method to fetch dataset from `torchvision.datasets`
def fetch_torchvision_dataset(args, dataset_name, root, transforms):
    logger.info(f'[LOAD] [{dataset_name.upper()}] Fetching dataset!')
    
    # default arguments
    DEFAULT_ARGS = {'root': root, 'transform': None, 'download': True}
    
    if dataset_name in [
        'MNIST', 'FashionMNIST', 'QMNIST', 'KMNIST', 'EMNIST',\
        'CIFAR10', 'CIFAR100', 'USPS'
    ]:
        # configure arguments for training and test dataset
        train_args = DEFAULT_ARGS.copy()
        train_args['train'] = True
        train_args['transform'] = transforms[0]
        test_args = DEFAULT_ARGS.copy()
        test_args['train'] = False
        test_args['transform'] = transforms[1]

        # special case - EMNIST
        if dataset_name == 'EMNIST':
            train_args['split'] = 'byclass'
            test_args['split'] = 'byclass'

        # create raw train/test dataset
        raw_train = torchvision.datasets.__dict__[dataset_name](**train_args)
        raw_test = torchvision.datasets.__dict__[dataset_name](**test_args)
        raw_train = VisionClassificationDataset(raw_train, dataset_name.upper(), 'CLIENT')
        raw_test = VisionClassificationDataset(raw_test, dataset_name.upper(), 'SERVER')

        # adjust arguemtns : in_channels
        if 'CIFAR' in dataset_name:
            args.in_channels = 3
        else:
            args.in_channels = 1


    args.num_classes = len(torch.unique(torch.as_tensor(raw_train.dataset.targets)))  
    logger.info(f'[LOAD] [{dataset_name.upper()}] ...fetched dataset!')

    return raw_train, raw_test, args


