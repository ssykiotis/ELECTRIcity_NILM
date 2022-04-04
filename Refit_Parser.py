import os
import numpy  as np
import pandas as pd
from pathlib import Path
from   NILM_Dataset     import *
from   Pretrain_Dataset import *

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


        self.cutoff        =  [args.cutoff[appl]    for appl in ['Aggregate']+args.appliance_names]
        self.threshold     =  [args.threshold[appl] for appl in args.appliance_names]
        self.min_on        =  [args.min_on[appl]    for appl in args.appliance_names]
        self.min_off       =  [args.min_off[appl]   for appl in args.appliance_names]

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
                house_data = house_data[['Aggregate',self.appliance_names[0]]]
                house_data = house_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)

                if house_idx == self.house_indicies[0]:
                    entire_data = house_data
                    if len(self.house_indicies) == 1:
                        entire_data = entire_data.reset_index(drop=True)
                else:
                    entire_data = entire_data.append(house_data, ignore_index=True)


        entire_data                  = entire_data.dropna().copy()
        entire_data                  = entire_data[entire_data['Aggregate'] > 0] #remove negative values (possible mistakes)
        entire_data[entire_data < 5] = 0 #remove very low values
        entire_data                  = entire_data.clip([0] * len(entire_data.columns), self.cutoff, axis=1) # force values to be between 0 and cutoff

        return entire_data.values[:, 0], entire_data.values[:, 1]





    def compute_status(self,data):
        initial_status = data >= self.threshold[0]
        status_diff    = np.diff(initial_status)
        events_idx     = status_diff.nonzero()

        events_idx  = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx     = events_idx.reshape((-1, 2))
        on_events      = events_idx[:, 0].copy()
        off_events     = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events    = on_events[off_duration > self.min_off[0]]
            off_events   = off_events[np.roll(off_duration, -1) > self.min_off[0]]

            on_duration  = off_events - on_events
            on_events    = on_events[on_duration  >= self.min_on[0]]
            off_events   = off_events[on_duration >= self.min_on[0]]
            assert len(on_events) == len(off_events)

        temp_status = data.copy()
        temp_status[:] = 0
        for on, off in zip(on_events, off_events):
            temp_status[on: off] = 1
        status = temp_status

        return status    

    def get_train_datasets(self):
        val_end = int(self.val_size * len(self.x))
        
        val = NILMDataset(self.x[:val_end],
                          self.y[:val_end],
                          self.status[:val_end],
                          self.window_size,
                          self.window_size    #non-overlapping windows
                          )

        train = NILMDataset(self.x[val_end:],
                            self.y[val_end:],
                            self.status[val_end:],
                            self.window_size,
                            self.window_stride
                            )
        return train, val

    def get_pretrain_datasets(self, mask_prob=0.25):
        val_end = int(self.val_size * len(self.x))

        val  = NILMDataset(self.x[:val_end],
                           self.y[:val_end],
                           self.status[:val_end],
                           self.window_size,
                           self.window_size
                          )
        train = Pretrain_Dataset(self.x[val_end:],
                                 self.y[val_end:],
                                 self.status[val_end:],
                                 self.window_size,
                                 self.window_stride,
                                 mask_prob=mask_prob
                                )
        return train, val
