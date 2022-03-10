import os
import torch
from   torch import nn
from   NILM_Dataloader import *



class Trainer:
    def __init__(self,args,ds_parser,model):
        self.args             = args
        self.device           = args.device
        self.num_epochs       = args.num_epochs
        # self.ds_parser        = ds_parser
        self.model            = model.to(args.device)
        self.export_root      = args.export_root
        self.best_model_epoch = None

        self.cutoff      = torch.Tensor(args.cutoff[args.appliance_names[0]]    ).to(self.device)
        self.threshold   = torch.Tensor(args.threshold[args.appliance_names[0]] ).to(self.device)
        self.min_on      = torch.Tensor(args.min_on[args.appliance_names[0]]    ).to(self.device)
        self.min_off     = torch.Tensor(args.min_off[args.appliance_names[0]]   ).to(self.device)
        self.C0          = torch.Tensor(args.c0[args.appliance_names[0]]        ).to(self.device)
        self.tau         = args.tau


        if self.pretrain:
            dataloader = NILMDataloader(args, ds_parser, pretrain=True)
            self.pretrain_loader, self.pretrain_val_loader = dataloader.get_dataloaders()

        dataloader     = NILMDataloader(args, ds_parser, pretrain=False)
        self.train_loader, self.val_loader = dataloader.get_dataloaders()

        self.normalize   = args.normalize
        if self.normalize == 'mean':
            self.x_mean, self.x_std = ds_parser.x_mean,ds_parser.x_std
            self.x_mean = torch.Tensor(self.x_mean).to(self.device)
            self.x_std  = torch.Tensor(self.x_std ).to(self.device)  

        self.mse      = nn.MSELoss()
        self.kl       = nn.KLDivLoss(        reduction = 'batchmean')
        self.bceloss  = nn.BCEWithLogitsLoss(reduction = 'mean')
        self.margin   = nn.SoftMarginLoss()
        self.l1_on    = nn.L1Loss(reduction='sum')

        # per epoch
        self.train_metrics_dict = {
                                    'mae'      : [],
                                    'mre'      : [],
                                    'acc'      : [],
                                    'precision': [],
                                    'recall'   : [],
                                    'f1'       : []       
                            }
        # per validate() run
        self.val_metrics_dict = {
                                'mae'      : [],
                                'mre'      : [],
                                'acc'      : [],
                                'precision': [],
                                'recall'   : [],
                                'f1'       : []       
                            }
        # test set
        self.test_metrics_dict = {
                                'mae'      : [],
                                'mre'      : [],
                                'acc'      : [],
                                'precision': [],
                                'recall'   : [],
                                'f1'       : []       
                        }


        self.training_loss = []
        self.y_pred_curve, self.y_curve, self.s_pred_curve, self.status_curve = [], [], [], []


        



    def train(self):
        _, best_mre, best_acc, _, _, best_f1 = self.validate()
        self._save_state_dict()
        if self.pretrain:
            for epoch in range(self.pretrain_num_epochs):
                self.pretrain_one_epoch(epoch+1)

        self.model.pretrain = False
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch+1)
            mae, mre, acc, precision, recall, f1 = self.validate()
            self.update_metrics_dict(mae, mre, acc, precision, recall, f1, mode = 'train')

            if f1 + acc - mre > best_f1 + best_acc - best_mre:
                best_f1      = f1
                best_acc     = acc
                best_mre     = mre
                self.best_model_epoch = epoch
                self._save_state_dict()

    def pretrain_one_epoch(self,epoch):
        pass
    
    
    def train_one_epoch(self,epoch):
        pass

    def validate(self):
        pass


    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model.state_dict(), self.export_root.joinpath('best_acc_model.pth'))

    def update_metrics_dict(self,mae,mre,acc,precision,recall,f1, mode = 'val'):
        if mode=='train':
            self.train_metrics_dict['mae'      ].append(mae)
            self.train_metrics_dict['mre'      ].append(mre)
            self.train_metrics_dict['acc'      ].append(acc)
            self.train_metrics_dict['precision'].append(precision)
            self.train_metrics_dict['recall'   ].append(recall)
            self.train_metrics_dict['f1'       ].append(f1)
        elif mode=='val':
            self.val_metrics_dict['mae'      ].append(mae)
            self.val_metrics_dict['mre'      ].append(mre)
            self.val_metrics_dict['acc'      ].append(acc)
            self.val_metrics_dict['precision'].append(precision)
            self.val_metrics_dict['recall'   ].append(recall)
            self.val_metrics_dict['f1'       ].append(f1)
        else: 
            self.test_metrics_dict['mae'      ].append(mae)
            self.test_metrics_dict['mre'      ].append(mre)
            self.test_metrics_dict['acc'      ].append(acc)
            self.test_metrics_dict['precision'].append(precision)
            self.test_metrics_dict['recall'   ].append(recall)
            self.test_metrics_dict['f1'       ].append(f1) 




