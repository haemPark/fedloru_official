import copy
import torch
import torch.nn as nn
import inspect

from torch.optim import lr_scheduler

from src import MetricManager, lowrank_resnet_loader
from .utils import get_conv_index, decide_upper_bound


class FedHMClient():
    def __init__(self, args, training_set, test_set, id):
        super(FedHMClient, self).__init__()
        self.args = args
        self.test_set = test_set
        self.training_set = training_set
        self.id = id
        self.rank_factor = self.args.rank_factor

        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]
        
        self.lowrank_model = None
        
        self.train_dataloader = self._create_dataloader(dataset=self.training_set, shuffle=not self.args.no_shuffle)
        self.test_dataloader = self._create_dataloader(dataset=self.test_set, shuffle=False)

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
            shuffle=shuffle, 
            #num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )

    def _load_lowrank_model(self):
        if 'ResNet' in self.args.model_name:
            model = lowrank_resnet_loader(self.args, self.rank_factor)
        return model

    def update(self, pbar, device):
        # define metric manager and constants
        mm = MetricManager(self.args.eval_metrics)
        
        # define model to train
        self.lowrank_model.train()
        self.lowrank_model.to(device)

        # define optimizer
        optimizer = self.optim(self.lowrank_model.parameters(), **self._refine_optim_args(self.args))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.gamma)

        # train new model
        for epoch in range(self.args.E):
            for inputs, targets in self.train_dataloader:
                pbar.update(1)
                inputs, targets = inputs.to(device), targets.to(device)
                if inputs.size(dim=0) > 1: # when train inputs are non-empty
                    outputs = self.lowrank_model(inputs)
                    loss = self.criterion()(outputs, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.lowrank_model.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                    mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), epoch + 1)
            scheduler.step()
        else:
            self.lowrank_model.to('cpu')
        return mm.results

    @torch.inference_mode()
    def evaluate(self):
        if self.args.test_size == 0:
            print("!!!!!! NO TEST DATA : TEST_SIZE = 0")
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        device = self.args.device
        self.lowrank_model.eval()

        self.lowrank_model.to(device)

        for inputs, targets in self.test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if inputs.size(dim=0) > 1:
                outputs = self.lowrank_model(inputs)
                loss = self.criterion()(outputs, targets)

                mm.track(loss.item(), outputs, targets)
        else:
            self.lowrank_model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    def recover_lowrank_model(self, fullrank_model):

        # Make Low-rank & Full-rank model params
        lowrank_model_params = list(self.lowrank_model.state_dict().values())
        fullrank_model_params = list(fullrank_model.state_dict().items())

        # State-Dict for Recovered Model
        reload_state_dict = {}
        for (param_name, param) in fullrank_model_params:
            reload_state_dict[param_name] = torch.zeros_like(param.data)
            
        with torch.no_grad():
            # Tools for Recover
            index_map = get_conv_index(fullrank_model, self.lowrank_model, self.rank_factor, self.args.freeze_bn, self.args.model_name)
            decompose_name, skip_name, upper_bound = decide_upper_bound(self.rank_factor, self.args.freeze_bn, self.args.model_name)
                                        
            for index, (param_name, param) in enumerate(fullrank_model_params):
                if 'conv' in param_name \
                        and index not in range(0, upper_bound) \
                        and skip_name not in param_name:
                    model_u_weight, model_v_weight = lowrank_model_params[index_map[index]], \
                                                    lowrank_model_params[index_map[index] + 1]

                    u_weight = model_v_weight  # .view(dim1, rank)
                    v_weight = model_u_weight  # .view(rank, dim2)

                    combine_weights = torch.matmul(u_weight, v_weight)
                    combine_weights = combine_weights.reshape(param.shape[0], param.shape[2], param.shape[1],
                                                            param.shape[3]).permute(0, 2, 1, 3)
                    # combine_weights = combine_weights.view(param.shape)

                    assert combine_weights.size() == param.data.size()
                    reload_state_dict[param_name] = combine_weights
                elif 'classifier' in param_name:
                    reload_state_dict[param_name] = lowrank_model_params[index_map[index]]
                else:
                    reload_state_dict[param_name] = lowrank_model_params[index_map[index]]
        return reload_state_dict

    def download(self, model):
        self.lowrank_model = copy.deepcopy(model)

    def __len__(self):
        return len(self.train_dataloader)

    def __repr__(self):
        return f'CLIENT < {self.id} >'


