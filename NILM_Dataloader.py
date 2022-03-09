import torch.utils.data as data_utils

class NILMDataloader():
    def __init__(self, args, ds_parser,pretrain = False):
        self.args       = args
        self.mask_prob  = args.mask_prob
        self.batch_size = args.batch_size

        if pretrain:
            self.train_dataset, self.val_dataset = ds_parser.get_pretrain_datasets(mask_prob=self.mask_prob)
        else:
            self.train_dataset, self.val_dataset = ds_parser.get_train_datasets()

    def get_dataloaders(self):
        train_loader = self._get_loader(self.train_dataset)
        val_loader   = self._get_loader(self.val_dataset)
        return train_loader, val_loader

    def _get_loader(self, dataset):
        dataloader = data_utils.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           pin_memory=True
                                          )
        return dataloader
