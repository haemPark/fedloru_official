import copy
import torch
import inspect
import itertools

from torch.optim import lr_scheduler

from src import MetricManager

class FedLoRUClient():
    def __init__(self, args, training_set, test_set, id):
        super(FedLoRUClient, self).__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set
        self.id = id
        self.model = None

        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_dataloader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_dataloader = self._create_dataloader(self.test_set, shuffle=False)

    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args
    
    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0:
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_size=self.args.B, 
            shuffle=shuffle
        )
    
    def update(self, pbar, device):
        # define metric manager and constants
        mm = MetricManager(self.args.eval_metrics)
        # define model to train
        self.model.train()
        self.model.to(device)

        # define optimizer
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.gamma)
        
        # train new model
        for epoch in range(self.args.E):
            for inputs, targets in self.train_dataloader:
                pbar.update(1)
                inputs, targets = inputs.to(device), targets.to(device)
                if inputs.size(dim=0) > 1: # when train inputs are non-empty
                    outputs = self.model(inputs)
                    loss = self.criterion()(outputs, targets)
                    
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
        self.model.to(device)

        for inputs, targets in self.test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if inputs.size(dim=0) > 1:
                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                mm.track(loss.item(), outputs, targets)
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    def download(self, model):
        self.model = copy.deepcopy(model)

    def upload(self):
        return itertools.chain.from_iterable([self.model.named_parameters(), self.model.named_buffers()])

    def __len__(self):
        return len(self.training_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'