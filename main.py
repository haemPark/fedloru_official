'''
    Federated Learning with Low-Rank Updates
    Follow the work of Vaseline555
    @Follow vaseline555:
        https://github.com/vaseline555/Federated-Learning-in-PyTorch
'''

import os
import sys
import time
import torch
import argparse
import traceback

from importlib import import_module

from src import Range, set_logger, set_seed, load_dataset, load_model, algorithmTextConverter

def main(args):
    # set seed for reproductibility
    set_seed(args.seed)

    # load dataset
    server_dataset, client_datasets = load_dataset(args)
    
    if args.algorithm == 'fedhm':
        ALGORITHM_CLASS = import_module(f'src.algorithm.{args.algorithm}.main').__dict__[f'{algorithmTextConverter(args.algorithm)}']
        algorithm = ALGORITHM_CLASS(args, server_dataset, client_datasets)
        algorithm.train()        
    else:
        # build model
        model, args = load_model(args)
        params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params before training: {params_before / 1_000_000:.2f}M")
    
        ALGORITHM_CLASS = import_module(f'src.algorithm.{args.algorithm}.main').__dict__[f'{algorithmTextConverter(args.algorithm)}']
        algorithm = ALGORITHM_CLASS(args, model, server_dataset, client_datasets)
        algorithm.train()


if __name__ == "__main__":
    # parse user inputs as arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    #####################
    # Default arguments #
    #####################
    parser.add_argument('--exp_name', help='name of the experiment', type=str, required=True)
    parser.add_argument('--seed', help='global random seed', type=int, default=3333)
    parser.add_argument('--device', help='device to use; `cpu`, `cuda`, `cuda:GPU_NUMBER`', type=str, default='cpu')    
    parser.add_argument('--data_path', help='path to save & read raw data', type=str, default='./data')
    parser.add_argument('--log_path', help='path to save logs', type=str, default='./log')
    parser.add_argument('--result_path', help='path to save results', type=str, default='./result')

    #####################
    # Dataset arguments #
    #####################
    # dataset configuration arguments
    parser.add_argument('--dataset', help='''name of dataset to use for an experiment (NOTE: case sensitive)
    - image classification datasets in `torchvision.datasets`,
    - text classification datasets in `torchtext.datasets`,
    - LEAF benchmarks [ FEMNIST | Sent140 | Shakespeare | CelebA | Reddit ],
    - among [ TinyImageNet | CINIC10 | SpeechCommands | BeerReviewsA | BeerReviewsL | Heart | Adult | Cover | GLEAM ]
    ''', type=str, required=True)
    parser.add_argument('--test_size', help='a fraction of local hold-out dataset for evaluation (-1 for assigning pre-defined test split as local holdout set)', type=float, choices=[Range(-1, 1.)], default=0.2)
    parser.add_argument('--num_workers', help='number of workers for dataloader', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true', default=False)

    # data augmentation arguments
    parser.add_argument('--resize', help='resize input images (using `torchvision.transforms.Resize`)', type=int, default=64)
    parser.add_argument('--crop', help='crop input images (using `torchvision.transforms.CenterCrop` (for evaluation) and `torchvision.transforms.RandomCrop` (for training))', type=int, default=32)
    parser.add_argument('--imnorm', help='normalize channels with mean 0.5 and standard deviation 0.5 (using `torchvision.transforms.Normalize`, if passed)', action='store_true')
    parser.add_argument('--randrot', help='randomly rotate input (using `torchvision.transforms.RandomRotation`)', type=int, default=None)
    parser.add_argument('--randhf', help='randomly flip input horizontaly (using `torchvision.transforms.RandomHorizontalFlip`)', type=float, choices=[Range(0., 1.)], default=0.5)
    parser.add_argument('--randvf', help='randomly flip input vertically (using `torchvision.transforms.RandomVerticalFlip`)', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randjit', help='randomly change the brightness and contrast (using `torchvision.transforms.ColorJitter`)', type=float, choices=[Range(0., 1.)], default=0.5)

    # statistical heterogeneity simulation arguments
    parser.add_argument('--split_type', help='''type of data split scenario
    - `iid`: statistically homogeneous setting,
    - `unbalanced`: unbalanced in sample counts across clients,
    - `patho`: pathological non-IID split scenario proposed in (McMahan et al., 2016),
    - `diri`: Dirichlet distribution-based split scenario proposed in (Hsu et al., 2019),
    - `pre`: pre-defined data split scenario
    ''', type=str, choices=['iid', 'unbalanced', 'patho', 'diri', 'pre'], default='iid')
    parser.add_argument('--mincls', help='the minimum number of distinct classes per client (valid only if `split_type` is `patho` or `diri`)', type=int, default=10)
    parser.add_argument('--min_samples', help='the minimum number of samples per client (valid only if `split_type` is `patho` or `diri`)', type=int, default=500)
    parser.add_argument('--cncntrtn', help='a concentration parameter for Dirichlet distribution (valid only if `split_type` is `diri`)', type=float, default=0.1)

    ###################
    # Model arguments #
    ###################
    parser.add_argument('--model_name', help='a model to be used (NOTE: case sensitive)', type=str,
        choices=[
            'LeNet', 'ResNet10', 'ResNet18', 'ResNet34'
        ],
        required=True
    )
    parser.add_argument('--hidden_size', help='hidden channel size for vision models, or hidden dimension of language models', type=int, default=64)
    parser.add_argument('--dropout', help='dropout rate', type=float, choices=[Range(0., 1.)], default=0.1)
    parser.add_argument('--init_type', help='weight initialization method', type=str, default='xavier', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'truncnorm', 'none'])
    parser.add_argument('--init_gain', help='magnitude of variance used for weight initialization', type=float, default=1.0)

    ######################
    # Learning arguments #
    ######################
    parser.add_argument('--algorithm', help='federated learning algorithm to be used', type=str,
        choices=['fedavg', 'fedloru', 'fedhm'],
        required=True
    )
    parser.add_argument('--eval_every', help='frequency of the evaluation (i.e., evaluate peformance of a model every `eval_every` round)', type=int, default=1)
    parser.add_argument('--eval_metrics', help='metric(s) used for evaluation', type=str,
        choices=[
            'acc1', 'acc5', 'auroc', 'auprc', 'youdenj', 'f1', 'precision', 'recall',
            'seqacc', 'mse', 'mae', 'mape', 'rmse', 'r2', 'd2'
        ], nargs='+', default=["acc1"]
    )
    parser.add_argument('--eval_type', help='''the evaluation type of a model trained from FL algorithm
    - `local`: evaluation of personalization model on local hold-out dataset  (i.e., evaluate personalized models using each client\'s local evaluation set)
    - `global`: evaluation of a global model on global hold-out dataset (i.e., evaluate the global model using separate holdout dataset located at the server)
    - 'both': combination of `local` and `global` setting
    ''', type=str,
        choices=['local', 'global', 'both'],
        default='global'
    )
    parser.add_argument('--K', help='number of total cilents participating in federated training', type=int, default=20)
    parser.add_argument('--R', help='number of total federated learning rounds', type=int, default=100)
    parser.add_argument('--C', help='sampling fraction of clients per round (full participation when 0 is passed)', type=float, choices=[Range(0., 1.)], default=0.1)
    parser.add_argument('--E', help='number of local epochs', type=int, default=1)
    parser.add_argument('--B', help='local batch size (full-batch training when zero is passed)', type=int, default=32)
    parser.add_argument('--beta1', help='server momentum factor', type=float, choices=[Range(0., 1.)], default=0.)
    parser.add_argument('--server_lr', help='server learning rate', type=float, choices=[Range(0., 1.)], default=1.0)

    # optimization arguments
    parser.add_argument('--optimizer', help='type of optimization method (NOTE: should be a sub-module of `torch.optim`, thus case-sensitive)', type=str, default='SGD')
    parser.add_argument('--max_grad_norm', help='a constant required for gradient clipping', type=float, choices=[Range(0., float('inf'))], default=0.1)
    parser.add_argument('--weight_decay', help='weight decay (L2 penalty)', type=float, choices=[Range(0., 1.)], default=1e-4)
    parser.add_argument('--momentum', help='momentum factor', type=float, choices=[Range(0., 1.)], default=0.9)
    parser.add_argument('--lr', help='learning rate for local updates in each client', type=float, choices=[Range(0., 100.)], default=0.1, required=True)
    parser.add_argument('--lr_decay', help='decay rate of learning rate', type=float, choices=[Range(0., 1.)], default=0.99)
    parser.add_argument('--lr_decay_step', help='intervals of learning rate decay', type=int, default=1)
    parser.add_argument('--criterion', help='objective function (NOTE: should be a submodule of `torch.nn`, thus case-sensitive)', type=str, default='CrossEntropyLoss')

    # optimization - cosine annealing warmup arguments
    parser.add_argument('--min_lr', help='minimum learning rate for cosine annealing', type=float, default=0.001)
    parser.add_argument('--cycle_mult', help='decay of cycle', type=float, default=1.0)
    parser.add_argument('--warmup_steps', help='warmup steps for annealing', type=int, default=0)
    parser.add_argument('--cos_gamma', help='decay of maximum cos annealing lr', type=float, default=1.0)
    
    # low-rank update arguments
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--accumulate_and_reinit", type=str, default="500")
    parser.add_argument("--accumulate", type=str, default="500")
    parser.add_argument("--rank_factor", type=float, default=2) # for FedHM
    parser.add_argument("--freeze_bn", type=bool, default=True)

    # parse arguments
    args = parser.parse_args()
    args.first_cycle_steps = args.R
    args.accumulate_and_reinit = list(map(int, args.accumulate_and_reinit.split(',')))
    args.accumulate = list(map(int, args.accumulate.split(',')))

    ########################
    # Making Path for Logs #
    ########################

    # make path for saving losses & metrics & models
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.result_path = os.path.join(args.result_path, f'{args.exp_name}_{curr_time}')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # make path for saving logs
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # initialize logger
    set_logger(f'{args.log_path}/{args.exp_name}_{curr_time}.log', args)

    # run main program
    torch.autograd.set_detect_anomaly(True)
    try:
        main(args)
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)