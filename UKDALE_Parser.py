import numpy            as     np
import pandas           as     pd
from   pathlib          import Path
from   collections      import defaultdict
from   NILM_Dataset     import *
from   Pretrain_Dataset import *


class UK_Dale_Parser:
    
    def __init__(self,args, stats = None):
        self.data_location   = args.ukdale_location
        self.house_indicies  = args.house_indicies
        self.appliance_names = args.appliance_names
        self.sampling        = args.sampling
        self.normalize       = args.normalize
        
        self.cutoff        =  [args.cutoff[appl]    for appl in ['aggregate']+args.appliance_names]
        self.threshold     =  [args.threshold[appl] for appl in args.appliance_names]
        self.min_on        =  [args.min_on[appl]    for appl in args.appliance_names]
        self.min_off       =  [args.min_off[appl]   for appl in args.appliance_names]

        self.val_size      =  args.validation_size
        self.window_size   =  args.window_size
        self.window_stride =  args.window_stride

        
        self.x, self.y     = self.load_data()
        
        # if self.normalize == 'mean':
        #     if stats is None:
        #         self.x_mean = np.mean(self.x)
        #         self.x_std  = np.std(self.x)
        #     else:
        #         self.x_mean,self.x_std = stats
        #     self.x = (self.x - self.x_mean) / self.x_std
        # elif self.normalize == 'minmax':
        #     if stats is None:
        #         self.x_min = min(self.x)
        #         self.x_max = max(self.x)
        #     else:
        #         self.x_min,self.x_max = stats
        #     self.x = (self.x - self.x_min)/(self.x_max-self.x_min)
            
        self.status = self.compute_status(self.y)
    
    def load_data(self):
        
        for appliance in self.appliance_names:
            assert appliance in ['dishwasher', 'fridge','microwave', 'washing_machine', 'kettle']
        
        for house_id in self.house_indicies:
            assert house_id in [1, 2, 3, 4, 5]
        
        directory = Path(self.data_location)
        
        for house_id in self.house_indicies:
            house_folder = directory.joinpath('house_' + str(house_id))
            house_label  = pd.read_csv(house_folder.joinpath('labels.dat'),    sep=' ', header=None)    
            house_data   = pd.read_csv(house_folder.joinpath('channel_1.dat'), sep=' ', header=None) #aggregate
            
            #read aggregate data and resample
            house_data.columns = ['time','aggregate']
            house_data['time'] = pd.to_datetime(house_data['time'], unit = 's')
            house_data         = house_data.set_index('time').resample(self.sampling).mean().fillna(method='ffill', limit=30)
            
            appliance_list = house_label.iloc[:, 1].values
            app_index_dict = defaultdict(list)
            
            #find if device exists in house and create a dictionary that contains the channel names
            for appliance in self.appliance_names:
                try:
                    idx = appliance_list.tolist().index(appliance)
                    app_index_dict[appliance].append(idx+1)
                except ValueError:
                    app_index_dict[appliance].append(-1)
                
            #if no devices found in house, remove the house and move to the next
            if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                self.house_indicies.remove(house_id)
                continue
                
            #Read appliance data and merge  
            for appliance in self.appliance_names:
                channel_idx = app_index_dict[appliance][0]
                if channel_idx == -1:
                    house_data.insert(len(house_data.columns), appliance, np.zeros(len(house_data)))
                else:
                    channel_path      = house_folder.joinpath('channel_' + str(channel_idx) + '.dat')
                    appl_data         = pd.read_csv(channel_path, sep = ' ', header = None)
                    appl_data.columns = ['time',appliance]
                    appl_data['time'] = pd.to_datetime(appl_data['time'],unit = 's')          
                    appl_data         = appl_data.set_index('time').resample(self.sampling).mean().fillna(method = 'ffill', limit = 30)   
                    house_data        = pd.merge(house_data, appl_data, how='inner', on='time')
                                            
            if house_id == self.house_indicies[0]:
                entire_data = house_data
                if len(self.house_indicies) == 1:
                    entire_data = entire_data.reset_index(drop=True)
            else:
                entire_data = entire_data.append(house_data, ignore_index=True)
                
        entire_data                  = entire_data.dropna().copy()
        entire_data                  = entire_data[entire_data['aggregate'] > 0] #remove negative values (possible mistakes)
        entire_data[entire_data < 5] = 0 #remove very low values
        entire_data                  = entire_data.clip([0] * len(entire_data.columns), self.cutoff, axis=1) # force values to be between 0 and cutoff
        
        return entire_data.values[:, 0], entire_data.values[:, 1]

    def compute_status(self, data):

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
        
        train = NILMDataset(self.x[val_end:],
                            self.y[val_end:],
                            self.status[val_end:],
                            self.window_size,
                            self.window_stride)
        
        val   = NILMDataset(self.x[:val_end],
                            self.y[:val_end],
                            self.status[:val_end],
                            self.window_size,
                            self.window_size) #non-overlapping windows

        return train, val

    def get_pretrain_datasets(self, mask_prob=0.25):
        val_end = int(self.val_size * len(self.x))

        val     = NILMDataset(self.x[:val_end],
                               self.y[:val_end],
                               self.status[:val_end],
                               self.window_size,
                               self.window_size
                             )
        train   = Pretrain_Dataset(self.x[val_end:],
                                   self.y[val_end:],
                                   self.status[val_end:],
                                   self.window_size,
                                   self.window_stride,
                                   mask_prob=mask_prob
                                   )
        return train, val        