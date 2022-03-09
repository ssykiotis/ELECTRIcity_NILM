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

    parser.add_argument('--dataset_code',     type = str,   default = 'uk_dale', choices=['redd_lf', 'uk_dale'])
    parser.add_argument('--house_indicies',   type = list,  default = [1, 2, 3, 4, 5])


    # REDD Dataset appliance names:    'refrigerator', 'washer_dryer',   'microwave','dishwasher'
    # UK Dale Dataset appliance names: 'fridge',       'washing_machine','microwave','dishwasher','kettle'
    parser.add_argument('--appliance_names',  type = list,  default = ['washing_machine'])




def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False  
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)   


# if args.dataset_code == 'redd_lf':
#     args.cutoff = {
#         'aggregate'   : 6000,
#         'refrigerator': 400,
#         'washer_dryer': 3500,
#         'microwave'   : 1800,
#         'dishwasher'  : 1200
#     }
#     args.threshold = {
#         'refrigerator': 50,
#         'washer_dryer': 20,
#         'microwave'   : 200,
#         'dishwasher'  : 10
#     }
#     args.min_on = {
#         'refrigerator': 10,
#         'washer_dryer': 300,
#         'microwave'   : 2,
#         'dishwasher'  : 300
#     }
#     args.min_off = {
#         'refrigerator': 2,
#         'washer_dryer': 26,
#         'microwave'   : 5,
#         'dishwasher'  : 300
#     }
#     args.c0 = {
#         'refrigerator': 1e-6,
#         'washer_dryer': 0.001,
#         'microwave'   : 1.,
#         'dishwasher'  : 1.
#     }
# elif args.dataset_code == 'uk_dale':    
#     args.cutoff = {
#         'aggregate'      : 6000,
#         'kettle'         : 3100,
#         'fridge'         : 300,
#         'washing_machine': 2500,
#         'microwave'      : 3000,
#         'dishwasher'     : 2500
#     }
#     args.threshold = {
#         'kettle'         : 2000,
#         'fridge'         : 50,
#         'washing_machine': 20,
#         'microwave'      : 200,
#         'dishwasher'     : 10
#     }
#     #multiply by 6 for seconds
#     args.min_on = {
#         'kettle'         : 2,
#         'fridge'         : 10,
#         'washing_machine': 300,
#         'microwave'      : 2,
#         'dishwasher'     : 300
#     }
#     #multiply by 6 for seconds
#     args.min_off = {
#         'kettle'         : 0,
#         'fridge'         : 2,
#         'washing_machine': 26,
#         'microwave'      : 5,
#         'dishwasher'     : 300
#     }
#     args.c0 = {
#         'kettle'         : 1.,
#         'fridge'         : 1e-6,
#         'washing_machine': 0.01,
#         'microwave'      : 1.,
#         'dishwasher'     : 1.
#     }