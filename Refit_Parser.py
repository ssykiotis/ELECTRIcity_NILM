import os
import numpy  as np
import pandas as pd
from pathlib import Path



class Refit_Parser:

    def __init__(self, args, stats = None):
        self.dataset_location = args.refit_location
        assert 'Data','Labels' in os.listdir(self.dataset_location);'Incorrect Folder Structure'
        self.data_location    = Path(args.refit_location).joinpath('Data')
        self.labels_location  = Path(args.refit_location).joinpath('Labels')

        self.appliance_names  = args.appliance_names
        self.sampling         = args.sampling
        self.normalize        = args.normalize

        self.house_indicies  = args.house_indicies


        # self.cutoff        =  [args.cutoff[appl]    for appl in ['aggregate']+args.appliance_names]
        # self.threshold     =  [args.threshold[appl] for appl in args.appliance_names]
        # self.min_on        =  [args.min_on[appl]    for appl in args.appliance_names]
        # self.min_off       =  [args.min_off[appl]   for appl in args.appliance_names]


        self.val_size      =  args.validation_size
        self.window_size   =  args.window_size
        self.window_stride =  args.window_stride

        
        self.x, self.y     = self.load_data()

        if self.normalize == 'mean':
            if stats is None:
                self.x_mean = np.mean(self.x)
                self.x_std  = np.std(self.x)
            else:
                self.x_mean,self.x_std = stats
            self.x = (self.x - self.x_mean) / self.x_std

        self.status = self.compute_status(self.y)



    def load_data(self):
        for house_idx in self.house_indicies:
            filename  = 'House'+str(house_idx)+'.csv'
            labelname = 'House'+str(house_idx)+'.txt'
            house_data_loc = self.data_location/filename

            with open(self.labels_location/labelname) as f:
                house_labels = f.readlines()

            house_labels = ['Time'] + house_labels[0].split(',')

            if self.appliance_names[0] in house_labels:
                house_data = pd.read_csv(house_data_loc)
                house_data['Unix'] = pd.to_datetime(house_data['Unix'], unit = 's')

                house_data         = house_data.drop(labels = ['Time'],axis = 1)
                house_data.columns = house_labels
                house_data = house_data.set_index('Time')

                idx_to_drop = house_data[house_data['Issues']==1].index
                house_data = house_data.drop(index = idx_to_drop, axis = 0)
                house_data = house_data['Aggregate',self.appliance_names[0]]
                house_data = house_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)


        return None,None





    def compute_status(self,data):
        pass
