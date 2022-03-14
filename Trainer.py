import os
import torch
import json
import numpy               as np
import torch.optim         as optim
import torch.nn.functional as F

from   tqdm                import tqdm
from   torch               import nn
from   pathlib             import Path
from   NILM_Dataloader     import *
from   metrics             import *

class Trainer:
    def __init__(self,args,ds_parser,model):
        self.args                = args
        self.device              = args.device
        self.pretrain            = args.pretrain
        self.pretrain_num_epochs = args.pretrain_num_epochs
        self.num_epochs          = args.num_epochs
        self.model               = model.to(args.device)
        self.export_root         = Path(args.export_root).joinpath(args.dataset_code).joinpath(args.appliance_names[0])
        self.best_model_epoch    = None

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

        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=args.decay_step,
                                                          gamma=args.gamma
                                                         )

        self.normalize   = args.normalize
        if self.normalize == 'mean':
            self.x_mean, self.x_std = ds_parser.x_mean,ds_parser.x_std
            self.x_mean = torch.tensor(self.x_mean).to(self.device)
            self.x_std  = torch.tensor(self.x_std ).to(self.device)  

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
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.pretrain_loader)
        for _,batch in enumerate(tqdm_dataloader):
            x, y, status = [batch[i].to(self.device) for i in range(3)]
            self.optimizer.zero_grad()

            mask = (status >= 0)

            y_capped = y / self.cutoff

            logits, gen_out, logits_y, logits_status    = self.get_model_outputs(x,mask)

            logits_masked        = torch.masked_select(logits       , mask).view((-1))
            labels_masked        = torch.masked_select(y_capped     , mask).view((-1))
            # status_masked        = torch.masked_select(status       , mask).view((-1))
            # logits_status_masked = torch.masked_select(logits_status, mask).view((-1))
            gen_out = gen_out.view(-1)

            mask = mask.view(-1).type(torch.DoubleTensor).to(self.device)

            total_loss = self.loss_fn_pretrain(logits_masked,labels_masked,gen_out,mask)
            
            total_loss.backward()
            self.optimizer.step()

            loss_values.append(total_loss.item())
            average_loss = np.mean(np.array(loss_values))
            self.training_loss.append(average_loss)
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()
    
    def train_one_epoch(self,epoch):
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for _,batch in enumerate(tqdm_dataloader):
            x, y, status = [batch[i].to(self.device) for i in range(3)]
            self.optimizer.zero_grad()
            y_capped = y / self.cutoff

            logits,_, logits_y, logits_status = self.get_model_outputs(x)
            total_loss                      = self.loss_fn_train(logits,y_capped,logits_status,status)
            
            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())
            average_loss = np.mean(np.array(loss_values))
            self.training_loss.append(average_loss)
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

    def validate(self):
        self.model.eval()
        self.val_metrics_dict = {
                            'mae'      : [],
                            'mre'      : [],
                            'acc'      : [],
                            'precision': [],
                            'recall'   : [],
                            'f1'       : []       
                            }

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for _,batch in enumerate(tqdm_dataloader):
                x, y, status = [batch[i].to(self.device) for i in range(3)]  
                
                y_capped = y / self.cutoff

                _,_, logits_y, logits_status = self.get_model_outputs(x)
                logits_y                        = logits_y * logits_status

                acc,precision,recall,f1         = acc_precision_recall_f1_score(logits_status,status)
                mae, mre                        = regression_errors(logits_y, y_capped)
                self.update_metrics_dict(mae,mre,acc,precision,recall,f1,mode = 'val')

                acc_mean     = np.mean(np.concatenate(self.val_metrics_dict['acc']).reshape(-1))
                f1_mean      = np.mean(np.concatenate(self.val_metrics_dict['f1'] ).reshape(-1))
                mre_mean     = np.mean(np.concatenate(self.val_metrics_dict['mre']).reshape(-1))
                tqdm_dataloader.set_description('Validation, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(mre_mean, acc_mean, f1_mean))

        return [np.array(np.concatenate(v)).mean(axis=0) for v in self.val_metrics_dict.values()]

    def test(self,test_loader):
        self._load_best_model()
        self.model.eval()
        y_pred_curve, y_curve,s_pred_curve,status_curve = [], [], [], []

        with torch.no_grad():
            tqdm_dataloader = tqdm(test_loader)
            for _, batch in enumerate(tqdm_dataloader):
                x, y, status = [batch[i].to(self.device) for i in range(3)] 
        
                y_capped = y / self.cutoff

                _,_, logits_y, logits_status = self.get_model_outputs(x)
                logits_y                        = logits_y * logits_status

                acc,precision,recall,f1         = acc_precision_recall_f1_score(logits_status,status)
                mae, mre                        = regression_errors(logits_y, y_capped)
                self.update_metrics_dict(mae,mre,acc,precision,recall,f1, mode = 'test')

                acc_mean     = np.mean(np.concatenate(self.test_metrics_dict['acc']).reshape(-1))
                f1_mean      = np.mean(np.concatenate(self.test_metrics_dict['f1'] ).reshape(-1))
                mre_mean     = np.mean(np.concatenate(self.test_metrics_dict['mre']).reshape(-1))
                tqdm_dataloader.set_description('Test, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(mre_mean, acc_mean, f1_mean))

                y_pred_curve.append(logits_y.detach().cpu().numpy().squeeze())
                y_curve.append(     y.detach(       ).cpu().numpy().squeeze())
                s_pred_curve.append(logits_status.detach().cpu().numpy().squeeze())
                status_curve.append(status.detach().cpu().numpy().squeeze())
            
            self.y_pred_curve  = np.concatenate(y_pred_curve).reshape(1,-1)
            self.y_curve       = np.concatenate(y_curve     ).reshape(1,-1)
            self.s_pred_curve  = np.concatenate(s_pred_curve).reshape(1,-1)
            self.status_curve  = np.concatenate(status_curve).reshape(1,-1)


        self._save_result({'gt': self.y_curve.tolist(),'pred': self.y_pred_curve.tolist()}, 'test_result.json')
        mre, mae = regression_errors(self.y_pred_curve, self.y_curve)
        acc, precision,recall, f1 = acc_precision_recall_f1_score(self.s_pred_curve, self.status_curve)

        return mre,mae,acc, precision,recall, f1

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

    def get_model_outputs(self, x,mask=None):

        logits, gen_out = self.model(x,mask)
        logits_y        = self.cutoff_energy(logits*self.cutoff)
        logits_status   = self.compute_status(logits_y)
    
        return logits, gen_out, logits_y, logits_status

    def cutoff_energy(self,data):
        data[data<5] = 0
        data = torch.min(data,self.cutoff.double())
        return data
        
    def compute_status(self,data):
        status = (data>=self.threshold)*1
        return status

    def _create_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay        = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if self.args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        elif self.args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=self.args.lr)
        elif self.args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=self.args.momentum)
        else:
            raise ValueError

    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model.state_dict(), self.export_root.joinpath('best_acc_model.pth'))

    def _load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(self.export_root.joinpath('best_acc_model.pth')))
            self.model.to(self.device)
        except:
            print('Failed to load best model, continue testing with current model...')
    
    def _save_result(self,data,filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        filepath = Path(self.export_root).joinpath(filename)
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)

    def loss_fn_gen(self,logits_masked,labels_masked):
        mse_arg_1 = logits_masked.contiguous().view(-1).double()
        mse_arg_2 = labels_masked.contiguous().view(-1).double()
        kl_arg_1  = torch.log(F.softmax(logits_masked.squeeze()/self.tau, dim=-1) + 1e-9)
        kl_arg_2  = F.softmax(labels_masked.squeeze()/self.tau, dim=-1)
        mse_loss = self.mse(mse_arg_1,mse_arg_2)
        kl_loss  = self.kl(kl_arg_1,kl_arg_2)
        loss = mse_loss + kl_loss
        return loss
        
    def loss_fn_disc(self,gen_out,mask):
        return self.bceloss(gen_out,mask)

    def loss_fn_pretrain(self,logits_masked,labels_masked,gen_out,mask):

        gen_loss  = self.loss_fn_gen(logits_masked,labels_masked)
        disc_loss = self.loss_fn_disc(gen_out,mask)
        return gen_loss + disc_loss 
        
    def loss_fn_train(self,logits,labels,logits_status,status):
        kl_arg_1     = torch.log(F.softmax(logits.squeeze() / 0.1, dim=-1) + 1e-9)
        kl_arg_2     = F.softmax(labels.squeeze() / 0.1, dim=-1)
        mse_arg_1    = logits.contiguous().view(-1).double()
        mse_arg_2    = labels.contiguous().view(-1).double()
        margin_arg_1 = (logits_status * 2 - 1).contiguous().view(-1).double()
        margin_arg_2 = (status * 2 - 1).contiguous().view(-1).double()

        kl_loss      = self.kl( kl_arg_1,  kl_arg_2)
        mse_loss     = self.mse(mse_arg_1, mse_arg_2)
        margin_loss  = self.margin(margin_arg_1, margin_arg_2)

        total_loss   = kl_loss + mse_loss + margin_loss
        
        on_mask = ((status == 1) + (status != logits_status.reshape(status.shape))) >= 1
        if on_mask.sum() > 0:
            total_size  = torch.tensor(on_mask.shape).prod()
            logits_on   = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
            labels_on   = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
            loss_l1_on  = self.l1_on(logits_on.contiguous().view(-1), labels_on.contiguous().view(-1))
            total_loss += self.C0 * loss_l1_on / total_size
        return total_loss

