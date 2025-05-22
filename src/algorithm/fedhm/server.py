import copy
import torch
import logging

from src import init_weights, fullrank_resnet_loader, lowrank_resnet_loader
from .utils import split_conv

logger = logging.getLogger(__name__)

class FedHMServer():
    def __init__(self, args, server_dataset):
        self.args = args
        
        self.round = 0 # round indicator
        self.rank_factor = self.args.rank_factor

        if self.args.eval_type != 'local':  # global holdout set for central evaluation
            self.server_dataset = server_dataset
        self.fullrank_model = self._model_loader()
        self.fullrank_model = self._init_model(self.fullrank_model)

    def _model_loader(self):
        if 'ResNet' in self.args.model_name:
            model = fullrank_resnet_loader(self.args)
        return model
    
    def _load_lowrank_model(self):
        if 'ResNet' in self.args.model_name:
            if self.rank_factor > 1:
                lowrank_model = lowrank_resnet_loader(self.args, self.rank_factor)
            else:
                lowrank_model = fullrank_resnet_loader(self.args)
        return lowrank_model

    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a global model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; (Initialization type: {self.args.init_type.upper()}))!')
        return model
    
    def get_factorized_lowrank_state_dict(self, rank_factor):
        lowrank_model = self._load_lowrank_model()
        if rank_factor > 1:
            assert lowrank_model.rank_factor == rank_factor

        with torch.no_grad():
            if rank_factor > 1:
                if 'ResNet' in self.args.model_name:
                    lowrank_state_dict = split_conv(
                        fullrank_model=self.fullrank_model,
                        lowrank_model=lowrank_model,
                        rank_factor=rank_factor,
                        freeze_bn=self.args.freeze_bn,
                        model_name=self.args.model_name
                    )
                else:
                    lowrank_state_dict = self.fullrank_model.state_dict()
            else:
                lowrank_state_dict = self.fullrank_model.state_dict()
        lowrank_model.load_state_dict(lowrank_state_dict)
        return lowrank_model
    

