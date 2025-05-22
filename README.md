# fedloru_official

# FedLoRU: Supplementary Material

This repository contains the supplementary material for the paper on the FedLoRU algorithm. It includes the source code and instructions to reproduce the experiments.

## Directory Structure

The project is organized as follows:

```bash
fedloru_official/
├── main.py # Main script to run experiments
├── requirements.txt # Python package dependencies
├── README.md # This file
└── src/ # Source code
    ├── __init__.py
    ├── utils.py
    ├── algorithm/ # Implementation of federated learning algorithms
    │   ├── fedavg/
    │   ├── fedhm/
    │   └── fedloru/ 
    ├── datasets/ # Data loading and processing utilities
    ├── metrics/ # Evaluation metrics
    ├── models/ # Model architectures
    └── optimizers/ # Optimizers and learning rate schedulers
```

## Requirements

The required Python packages are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

## How to Run

To run the FedLoRU algorithm, use the main.py script with the desired arguments.

For example, to run FedLoRU on CIFAR-100 with ResNet18:

```bash
python3 main.py --exp_name=fedloru_cifar100 --seed=3 --device=cuda --dataset=CIFAR100 --model_name=ResNet18 --algorithm=fedloru --split_type=iid --randhf=0.5 --randjit=0.5 --resize=64 --crop=32 --B=32 --K=100 --C=0.5 --E=5 --R=400 --lr=0.2 --eval_every=2 --r=128 --accumulate=50,100,150,200,250,300,350
```