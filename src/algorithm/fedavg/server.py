
import logging
from src import init_weights

logger = logging.getLogger(__name__)

class FedAvgServer():
    def __init__(self, args, server_dataset, model):
        super(FedAvgServer, self).__init__()
        self.args = args

        self.round = 0 # round indicator
        if self.args.eval_type != 'local':  # global holdout set for central evaluation
            self.server_dataset = server_dataset
        self.global_model = self._init_model(model) # global model initiailization


    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a global model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; (Initialization type: {self.args.init_type.upper()}))!')
        return model
        