import copy
import torch
import torch.nn as nn
import inspect
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim import lr_scheduler

from src import MetricManager

class FedAvgClient():
    def __init__(self, args, training_set, test_set, id):
        super(FedAvgClient, self).__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set
        self.id = id

        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_dataloader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_dataloader = self._create_dataloader(self.test_set, shuffle=False)
    
    # For Test
    def _model_distance(self, model1, model2):
        # Initialize distance
        distance = 0.0
        
        # Iterate through the parameters of both models
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            # Compute the L2 norm (Euclidean distance) between corresponding parameters
            distance += torch.norm(param1 - param2).item()
        
        return distance
    
    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args
    
    def _create_distributed_dataloader(self, dataset, shuffle):
        if self.args.B == 0:
            self.args.B = len(self.training_set)
        self.train_sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
        return torch.utils.data.DataLoader(
                dataset=dataset, 
                batch_size=int(self.args.B / self.args.world_size),
                shuffle=False,
                num_workers=int(self.args.num_workers / self.args.world_size),
                sampler=self.train_sampler
            )
    
    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0:
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_size=self.args.B, 
            shuffle=shuffle, 
            num_workers=self.args.num_workers,
            pin_memory=True
        )
    
    def dist_update(self, pbar, device):
        mp.spawn(self.update, nprocs=self.args.world_size, args=(pbar, device), join=True)
        # Ensure 'spawn' start method for multiprocessing
        
    def main_worker(self, gpu, pbar, device):
        print(gpu) 

    def update(self, pbar, device):        
        # define metric manager and constants
        mm = MetricManager(self.args.eval_metrics)
        
        # define model to train
        self.model.train()
        self.model.to(device)

        # define optimizer
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay_step)

        # train new model
        for epoch in range(self.args.E):
            for inputs, targets in self.train_dataloader:
                prev_model = copy.deepcopy(self.model)
                pbar.update(1)
                inputs, targets = inputs.to(device), targets.to(device)
                if inputs.size(dim=0) > 1: # when train inputs are non-empty
                    outputs = self.model(inputs)
                    # if self.args.model_name in ["ResNet18", "ResNet34"]:
                    #     outputs = outputs.logits
                    loss = self.criterion()(outputs, targets)
                    
                    # For FedProx
                    # if self.args.mu > 0 and epoch > 0:
                    #     # Add proximal term to loss (FedProx)
                    #     w_diff = torch.tensor(0., device=device)
                    #     for w, w_t in zip(self.model.parameters(), model_server.parameters()):
                    #         w_diff += torch.pow(torch.norm(w.data - w_t.data), 2)
                    #         #w.grad.data += self.args.mu * (w.data - w_t.data)
                    #         w.grad.data += self.args.mu * (w_t.data - w.data)
                    #     loss += self.args.mu / 2. * w_diff
                    optimizer.zero_grad()
                    loss.backward()
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                    mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), epoch + 1)
            scheduler.step()
        else:
            self.model.to('cpu')
        return mm.results
                    
    @torch.inference_mode()
    def evaluate(self):
        if self.args.test_size == 0:
            print("!!!!!! NO TEST DATA : TEST_SIZE = 0")
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        device = self.args.device
        self.model.eval()
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)

        self.model.to(device)

        for inputs, targets in self.test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if inputs.size(dim=0) > 1:
                outputs = self.model(inputs)
                # if self.args.model_name in ["ResNet18", "ResNet34"]:
                #     outputs = outputs.logits
                loss = self.criterion()(outputs, targets)

                mm.track(loss.item(), outputs, targets)
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    def download(self, model):
        self.model = copy.deepcopy(model)

    def __len__(self):
        return len(self.training_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'