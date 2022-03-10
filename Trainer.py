import torch
from NILM_Dataloader import *


class Trainer:
    def __init__(self,args,ds_parser,model):
        self.args             = args
        self.device           = args.device
        self.num_epochs       = args.num_epochs
        # self.ds_parser        = ds_parser
        self.model            = model.to(args.device)
        self.export_root      = args.export_root
        self.best_model_epoch = None

        self.cutoff      = torch.tensor(args.cutoff[args.appliance_names[0]]    ).to(self.device)
        self.threshold   = torch.tensor(args.threshold[args.appliance_names[0]] ).to(self.device)
        self.min_on      = torch.tensor(args.min_on[args.appliance_names[0]]    ).to(self.device)
        self.min_off     = torch.tensor(args.min_off[args.appliance_names[0]]   ).to(self.device)
        self.C0          = torch.tensor(args.c0[args.appliance_names[0]]        ).to(self.device)
        self.tau         = args.tau


        if self.pretrain:
            dataloader = NILMDataloader(args, ds_parser, pretrain=True)
            self.pretrain_loader, self.pretrain_val_loader = dataloader.get_dataloaders()

        dataloader     = NILMDataloader(args, ds_parser, pretrain=False)
        self.train_loader, self.val_loader = dataloader.get_dataloaders()

        



    def train(self):
        pass


    def pretrain_one_epoch(self,epoch):
        pass
    
    
    def train_one_epoch(self,epoch):
        pass

    def validate(self):
        pass




    # if args.normalize =='mean':
    #     stats = (ds_parser.xmean,ds_parser.std)
    # else:
    #     stats = (ds_parser.x_min,ds_parser.x_max)