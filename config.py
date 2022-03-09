import argparse
import torch 
import numpy as np
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--redd_location',   type = str, default = None)
    parser.add_argument('--ukdale_location', type = str, default = None)

    parser.add_argument('--seed',            type = int,   default = 0)
    parser.add_argument('--device',          type = str,   default = 'cpu' ,    choices=['cpu', 'cuda'])

    parser.add_argument('--dataset_code',    type = str,   default = 'redd_lf', choices=['redd_lf', 'uk_dale'])
    parser.add_argument('--house_indicies',  type = list,  default = [1, 2, 3, 4, 5])

    # REDD Dataset appliance names:    'refrigerator', 'washer_dryer',   'microwave','dishwasher'
    # UK Dale Dataset appliance names: 'fridge',       'washing_machine','microwave','dishwasher','kettle'
    parser.add_argument('--appliance_names', type = list,  default = ['washer_dryer'])

    parser.add_argument('--sampling',        type = str,   default = '6s')
    parser.add_argument('--normalize',       type = str,   default = 'mean',    choices=['mean', 'minmax'])

    parser.add_argument('--c0',              type = dict,  default = None)  #temperature value for objective function
    parser.add_argument('--cutoff',          type = dict,  default = None)
    parser.add_argument('--threshold',       type = dict,  default = None)
    parser.add_argument('--min_on',          type = dict,  default = None)
    parser.add_argument('--min_off',         type = dict,  default = None)

    parser.add_argument('--window_size',         type = int,   default = 480)
    parser.add_argument('--window_stride',       type = int,   default = 120)
    parser.add_argument('--validation_size',     type = float, default = 0.1)
    parser.add_argument('--batch_size',          type = int,   default = 64)

    
    args = parser.parse_args()

    # args.ukdale_location = 'data/uk_dale'
    # args.redd_location   = 'data/redd'

    #MAC
    args.ukdale_location = '/Volumes/WD_2TB/PhD Datasets/Cleaned/Energy/UK_Dale'
    args.redd_location   = '/Volumes/WD_2TB/PhD Datasets/Cleaned/Energy/REDD'

    #UBUNTU
    args.ukdale_location = ''
    args.redd_location   = ''

    args = update_preprocessing_parameters(args)
    if torch.cuda.is_available():
        args.device = 'cuda:0'

    return args



def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False  
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)   


def update_preprocessing_parameters(args):
    if args.dataset_code == 'redd_lf':
        args.cutoff = {
            'aggregate'   : 6000,
            'refrigerator': 400,
            'washer_dryer': 3500,
            'microwave'   : 1800,
            'dishwasher'  : 1200
        }
        args.threshold = {
            'refrigerator': 50,
            'washer_dryer': 20,
            'microwave'   : 200,
            'dishwasher'  : 10
        }
        args.min_on = {
            'refrigerator': 10,
            'washer_dryer': 300,
            'microwave'   : 2,
            'dishwasher'  : 300
        }
        args.min_off = {
            'refrigerator': 2,
            'washer_dryer': 26,
            'microwave'   : 5,
            'dishwasher'  : 300
        }
        args.c0 = {
            'refrigerator': 1e-6,
            'washer_dryer': 0.001,
            'microwave'   : 1.,
            'dishwasher'  : 1.
        }
    elif args.dataset_code == 'uk_dale':    
        args.cutoff = {
            'aggregate'      : 6000,
            'kettle'         : 3100,
            'fridge'         : 300,
            'washing_machine': 2500,
            'microwave'      : 3000,
            'dishwasher'     : 2500
        }
        args.threshold = {
            'kettle'         : 2000,
            'fridge'         : 50,
            'washing_machine': 20,
            'microwave'      : 200,
            'dishwasher'     : 10
        }
        #multiply by 6 for seconds
        args.min_on = {
            'kettle'         : 2,
            'fridge'         : 10,
            'washing_machine': 300,
            'microwave'      : 2,
            'dishwasher'     : 300
        }
        #multiply by 6 for seconds
        args.min_off = {
            'kettle'         : 0,
            'fridge'         : 2,
            'washing_machine': 26,
            'microwave'      : 5,
            'dishwasher'     : 300
        }
        args.c0 = {
            'kettle'         : 1.,
            'fridge'         : 1e-6,
            'washing_machine': 0.01,
            'microwave'      : 1.,
            'dishwasher'     : 1.
        }

    args.window_stride  = 120 if args.dataset_code == 'redd_lf' else 240
    args.house_indicies = [1, 2, 3, 4, 5, 6] if args.dataset_code == 'redd_lf' else [1,2,3,4,5]
    return args