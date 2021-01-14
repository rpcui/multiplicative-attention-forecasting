import numpy as np
import pandas as pd
import os
import datetime
import random
import pickle
import torch
from torch.autograd import Variable
import sys
from utils import augmentation, city_group_augmentation

import pdb

random.seed(1000)

class Reader(object):
    def __init__(self, data, label, stat_date, city_id, not_in_holiday, augmented_scale, config, phase):
        self.data = data
        self.label = label
        self.stat_date = stat_date
        self.city_id = city_id
        self.not_in_holiday = not_in_holiday
        
        self.augmented_scale = augmented_scale['scale']
        self.city_group = augmented_scale['group']
        
        self.phase = phase
        
        self.input_ndim = data.shape[1]
        self.output_ndim = label.shape[1]
        
        self.config = self._update_config(config)

        self.num_city = config['num_city']
        
        # add extra data for feature extraction from labels
        self.lw_range = config['lw_range'] + config['extra_length']
        self.up_range = config['up_range'] + config['extra_length']
        
        self.warm_start_for_train = config['warm_start_for_train']
        self.warm_start_for_valid = config['warm_start_for_valid'] + config['extra_length']

        if phase == 'train':
            start = config['start_%s' % phase]
            end   = config['end_%s'   % phase] - datetime.timedelta(days=self.lw_range)
            
            index_data = pd.DataFrame({'city_id': city_id.values,
                                       'start_date': stat_date.values})
            index_data = index_data[(index_data.start_date >= start) & 
                                    (index_data.start_date <= end)]
            
        elif phase == 'valid' or phase == 'test':
            start = config['start_%s' % phase] - datetime.timedelta(days=self.warm_start_for_valid)
            end   = config['end_%s'   % phase]
            
            city_list = list(set(self.city_id))
            index_data = pd.DataFrame({'city_id': city_list,
                                       'start_date': [start for _ in range(len(city_list))]})
        else:
            raise Exception('Invalid phase value!')
        
        self.start = start
        self.end   = end
        self.index_data = index_data.reset_index(drop=True)
        
        self.sample_num = self.index_data.shape[0]
        self.start_date = config['start_%s' % phase]
        self.end_date   = config['end_%s'   % phase]
        
        self.batch_size = config['batch_size']
        self.batch_num = int(np.ceil(self.sample_num / self.batch_size))
        self.index = list(range(self.sample_num))
        
        self.curr = 0
        if phase == 'train':
            random.shuffle(self.index)
        
    def _update_config(self, config):
        config['num_feature'] = self.input_ndim
        
        if 'history' in config['feat_collections']:        
            config['index_of_avg'], config['index_of_avg_by_weekday'] = [], []
            src_columns = list(self.data.columns)
            dst_columns = list(self.label.columns)[: self.output_ndim]

            for window_len in config['window_set_for_avg']:
                config['index_of_avg'].append([src_columns.index('avg_%s_of_%d_days' % (metric, window_len)) 
                                               for metric in dst_columns])
            for window_len in config['window_set_for_avg_by_weekday']:
                config['index_of_avg_by_weekday'].append([src_columns.index('avg_%s_of_%d_weekdays' % (metric, window_len))
                                                          for metric in dst_columns])

            config['num_hist_columns'] = sum(map(len, config['index_of_avg'])) + \
                                         sum(map(len, config['index_of_avg_by_weekday']))
        
        return config
    
    def iterate(self, return_info=False):
        indices = self.index[self.curr * self.batch_size: (self.curr + 1) * self.batch_size]
        city_ids    = self.index_data.loc[indices, 'city_id'].tolist()
        start_dates = self.index_data.loc[indices, 'start_date'].tolist()
        
        batch_size = len(indices)
        
        assert pd.isna(city_ids).sum() == 0, 'NaNs in index_data, indices = %s' % indices 
        
        if self.phase == 'train':
            length = np.random.randint(self.lw_range, self.up_range)
        else:
            length = (self.end - self.start).days + 1
            
        input = np.zeros((batch_size, length, self.input_ndim), dtype=np.float32)
        label = np.zeros((batch_size, length, self.output_ndim), dtype=np.float32)
        mask = np.zeros((batch_size, length), dtype=np.float32)
        time = np.zeros((length, batch_size), dtype=np.float32)
        not_in_holiday = np.ones((batch_size, length), dtype=np.float32)
        
        for i, (start_date, city_id) in enumerate(zip(start_dates, city_ids)):
            time_range = min(length, (self.config['end_%s' % self.phase] - start_date).days + 1)
            if time_range < length:
                start_date = self.config['end_%s' % self.phase] - datetime.timedelta(days=length - 1)
                start_dates[i] = start_date
                time_range = min(length, (self.config['end_%s' % self.phase] - start_date).days + 1)
                        
            ind = (self.stat_date >= start_date) &\
                  (self.stat_date <  start_date + datetime.timedelta(days=time_range)) &\
                  (self.city_id == city_id)
            time_start = (pd.to_datetime(start_date).date() -
                          self.config['temporal_origin']).days
            
            mask[i, : time_range] = 1.
            not_in_holiday[i, : time_range] = self.not_in_holiday[ind].values
            time[: time_range, i] = np.arange(time_start, time_start + time_range)  
            
            if self.phase == 'train' and random.random() < self.config['augment_prob']:
                group_key = random.choice(self.city_group[city_id])
                
                cities = sorted([int(x) for x in group_key.split('-')])
                tgt_scale = self.augmented_scale[group_key]
                
                src_data = np.zeros((length, len(cities), self.input_ndim), dtype=np.float32)
                src_label = np.zeros((length, len(cities), self.output_ndim), dtype=np.float32)
                src_scale = np.zeros((len(cities), self.output_ndim), dtype=np.float32)
                
                for k, cid in enumerate(cities):
                    ind = (self.stat_date >= start_date) &\
                          (self.stat_date <  start_date + datetime.timedelta(days=time_range)) &\
                          (self.city_id == cid)
                    seq_len = int(np.sum(ind))
                    if seq_len > 0:
                        src_data[-seq_len: , k, :] = self.data.loc[ind].values
                        src_label[-seq_len: , k, :] = self.label.loc[ind].values
                        src_scale[k, :] = self.augmented_scale[str(cid)]
                
                src_label = src_label * src_scale[None, :, :]

                tgt_data, tgt_label = city_group_augmentation(src_data, src_label, tgt_scale, self.config)
                input[i, : time_range, :] = tgt_data
                label[i, : time_range, :] = tgt_label
                
                city_ids[i] = cities
            
            else:
                input[i, : time_range, :] = self.data.loc[ind].values
                label[i, : time_range, :] = self.label.loc[ind].values 
                
            
        if self.phase == 'train':
            input, label = augmentation(input, label, self.config)
            
            city_onehot = np.zeros((batch_size, self.num_city), dtype=np.float32)
            for i, city_id in enumerate(city_ids):
                if type(city_id) == list:
                    city_onehot[i, :] = np.sum(np.arange(self.num_city) == np.array(city_id)[:, None], 
                                               axis=0) / float(len(city_id))
                else:
                    city_onehot[i, :] = (np.arange(self.num_city) == city_id)
            
        else:
            city_onehot = (np.arange(self.num_city) == np.array(city_ids)[:, None]).astype(np.float32)
            
        
        self.curr += 1
        if self.curr == self.batch_num:
            self.curr = 0
            if self.phase == 'train':
                random.shuffle(self.index)
        
        input = np.transpose(input, (1, 0, 2))
        label = np.transpose(label, (1, 0, 2))
        mask = np.transpose(mask, (1, 0))
        not_in_holiday = np.transpose(not_in_holiday, (1, 0))
        
        input = Variable(torch.Tensor(input)).cuda()
        label = Variable(torch.Tensor(label)).cuda()
        city_onehot = Variable(torch.Tensor(city_onehot)).cuda()
        mask = Variable(torch.Tensor(mask)).cuda()
        time = Variable(torch.Tensor(time)).cuda()
        not_in_holiday = Variable(torch.Tensor(not_in_holiday)).cuda()
        
        if return_info == True:
            return (input, label, city_onehot, mask, time), (not_in_holiday, city_ids, start_dates)
        else:
            return (input, label, city_onehot, mask, time)
        
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../utils/')
    from config import config
    from features import build_data
    
    data_raw = pd.read_csv('~/data_factory/city_day_features.csv', sep='\t')
    data_raw.columns = [col.replace('city_day_features_new.', '') for col in data_raw.columns]
    
    # prepare train/valid/test sets
    preload_data = 'database_%s.pkl' % config['start_test'].strftime('%Y%m%d')
    with open(preload_data, 'rb') as f:
        outputs = pickle.load(f)
    
    # load scales for data and labels
    with open('augmented_scale.pkl', 'rb') as f:
        load = pickle.load(f)
        
        augmented_scale = load['augmented_scale']
        scale = load['scale']
    
    data = outputs[0]
    label = outputs[1]
    stat_date = outputs[2]
    city_id = outputs[3]
    not_in_holiday = outputs[4]
    
    reader = Reader(data, 
                    label, 
                    stat_date, 
                    city_id, 
                    not_in_holiday, 
                    augmented_scale, 
                    config, 'train')
    
    for i in range(10000):
        print(i)
        (input, label, city_id, mask, time), (not_in_holiday, ids, start_dates) = reader.iterate(return_info=True)
        pdb.set_trace()
        if np.sum(np.isnan(label.cpu().data.numpy())) > 0 or np.sum(np.isnan(input.cpu().data.numpy())) > 0:
            print('error')
            pdb.set_trace()
    