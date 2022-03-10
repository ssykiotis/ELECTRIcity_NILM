import argparse
import torch 
import numpy as np
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--redd_location',      type = str, default = None)
    parser.add_argument('--ukdale_location',    type = str, default = None)
    parser.add_argument('--export_root',        type = str, default = 'results')


    parser.add_argument('--seed',               type = int,   default = 0)
    parser.add_argument('--device',             type = str,   default = 'cuda:0' ,    choices=['cpu', 'cuda'])

    parser.add_argument('--dataset_code',       type = str,   default = 'redd_lf', choices=['redd_lf', 'uk_dale'])
    parser.add_argument('--house_indicies',     type = list,  default = [1, 2, 3, 4, 5])

    # REDD Dataset appliance names:    'refrigerator', 'washer_dryer',   'microwave','dishwasher'
    # UK Dale Dataset appliance names: 'fridge',       'washing_machine','microwave','dishwasher','kettle'
    parser.add_argument('--appliance_names',    type = list,  default = ['washer_dryer'])

    parser.add_argument('--sampling',           type = str,   default = '6s')
    parser.add_argument('--normalize',          type = str,   default = 'mean',    choices=['mean', 'minmax'])

    parser.add_argument('--c0',                 type = dict,  default = None)  #temperature value for objective function
    parser.add_argument('--cutoff',             type = dict,  default = None)
    parser.add_argument('--threshold',          type = dict,  default = None)
    parser.add_argument('--min_on',             type = dict,  default = None)
    parser.add_argument('--min_off',            type = dict,  default = None)

    parser.add_argument('--window_size',         type = int,   default = 480)
    parser.add_argument('--window_stride',       type = int,   default = 120)
    parser.add_argument('--validation_size',     type = float, default = 0.1)
    parser.add_argument('--batch_size',          type = int,   default = 16)


    parser.add_argument('--output_size',         type = int,   default = 1)
    parser.add_argument('--drop_out',            type = float, default = 0.1)
    parser.add_argument('--hidden',              type = int,   default = 256)
    parser.add_argument('--heads',               type = int,   default = 2)
    parser.add_argument('--n_layers',            type = int,   default = 2)

    parser.add_argument('--pretrain',            type = bool,  default = True)
    parser.add_argument('--mask_prob',           type = float, default = 0.25)
    parser.add_argument('--pretrain_num_epochs', type = int,   default = 1)
    parser.add_argument('--num_epochs',          type = int,   default = 1)
    parser.add_argument('--tau',                 type = float, default = 0.1)


    parser.add_argument('--optimizer',           type = str,   default = 'adam',    choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr',                  type = float, default = 1e-4)
    parser.add_argument('--enable_lr_schedule',  type = bool,  default = False)
    parser.add_argument('--weight_decay',        type = float, default = 0.)
    parser.add_argument('--momentum',            type = float, default = None)
    parser.add_argument('--decay_step',          type = int,   default = 100)




    
    args = parser.parse_args()

    # args.ukdale_location = 'data/uk_dale'
    # args.redd_location   = 'data/redd'

    #MAC
    # args.ukdale_location = '/Volumes/WD_2TB/PhD Datasets/Cleaned/Energy/UK_Dale'
    # args.redd_location   = '/Volumes/WD_2TB/PhD Datasets/Cleaned/Energy/REDD'

    #UBUNTU
    args.ukdale_location = '/media/stavros/WD_2TB/PhD Datasets/Cleaned/Energy/UK_Dale'
    args.redd_location   = '/media/stavros/WD_2TB/PhD Datasets/Cleaned/Energy/REDD'

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