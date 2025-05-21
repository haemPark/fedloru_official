import torch
import logging
import transformers

# turn off unnecessary logging
transformers.logging.set_verbosity_error()

from .utils import Range, MetricManager, set_seed, algorithmTextConverter
from .datasets import load_dataset
from .optimizers import CosineAnnealingWarmupRestarts
from .models import load_model, fullrank_resnet_loader, lowrank_resnet_loader



# for logger initialization
def set_logger(path, args):
    # initialize logger
    logger = logging.getLogger(__name__)
    logging_format = logging.Formatter(
        fmt='[%(levelname)s] (%(asctime)s) %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p'
    )
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(path)
    
    stream_handler.setFormatter(logging_format)
    file_handler.setFormatter(logging_format)
    
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level=logging.INFO)
    
    welcome_message = """
    _______ _______ ______  _______  _____   _____  _______ _______ ______ 
    |______ |______ |     \ |______ |_____/ |_____|    |    |______ |     \\
    |       |______ |_____/ |______ |    \_ |     |    |    |______ |_____/
    |      
        |      _______  _____   _____  __   _ __ __ __   _ ______
        |      |______ |_____| |_____/ | \  |   |   | \  | |  ____        
        |_____ |______ |     | |    \_ |  \_| __|__ |  \_| |_____|     
                                                                    
                                                        @Follow vaseline555
    """
    logger.info(welcome_message)

# weights initialization
def init_weights(model, init_type, init_gain):
    """Initialize network weights.

    Args:
        model (torch.nn.Module): network to be initialized
        init_type (string): the name of an initialization method: normal | xavier | xavier_uniform | kaiming | orthogonal | none
        init_gain (float): scaling factor for normal, xavier and orthogonal

    Returns:
        model (torch.nn.Module): initialized model with `init_type` and `init_gain`
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Linear') == 0 or classname.find('Conv') == 0):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, mean=0., std=init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'truncnorm':
                torch.nn.init.trunc_normal_(m.weight.data, mean=0., std=init_gain)
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f'[ERROR] Initialization method {init_type} is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

