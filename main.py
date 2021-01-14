import os
import glog
import pandas as pd
import numpy as np
import pickle
import datetime
import torch
from torch.autograd import Variable
import pdb

from model import Model
# from model_wo_factor import Model
from reader import Reader

from utils import update_feature_using_prediction


import sys
sys.path.insert(0, '../util/')
from config import config
from features_global import build_data
from util import save_backup, save_database, evaluate, normalized_quantile_loss

pd.set_option('display.max_columns', 1000)

def load_database(config):
    raw_data = pd.read_csv('~/data_factory/city_day_features.csv', sep='\t')
    raw_data.columns = [col.replace('city_day_features_new.', '') for col in raw_data.columns]
    
    # prepare train/valid/test sets
    preload_data = 'database_%s.pkl' % config['start_test'].strftime('%Y%m%d')
    if os.path.exists(preload_data):
        with open(preload_data, 'rb') as f:
            outputs = pickle.load(f)
    else:
        outputs = build_data(raw_data, 
                             config,
                             return_scale=False)
        with open(preload_data, 'wb') as f:
            pickle.dump(outputs, f)
    
    # load scales for data and labels
    with open('augmented_scale.pkl', 'rb') as f:
        load = pickle.load(f)
        
        augmented_scale = load['augmented_scale']
        scale = load['scale']
    
    return raw_data, outputs, augmented_scale, scale

    
def test(model, data_set, scale, raw_data, config):
    fcst_len = config['forecast_length']
    extr_len = config['extra_length']
    
    preds = pd.DataFrame()
    
    for iter in range(data_set.batch_num):
        (input, label, city_onehot, mask, time), info = data_set.iterate(return_info=True)
        
        output, quantile_10, quantile_90 = label.clone(), label.clone(), label.clone()
        n_timestep = label.size()[0]
        
        not_in_holiday, city_ids, start_dates = info[0], info[1], info[2]
        pred, pred_10, pred_90, _ = model.infer(input, label, city_onehot, mask, time, return_loss=True)
        
        prediction = np.transpose(pred[-fcst_len: ], (1, 0, 2))  # (nb, length, num_out)
        
        quantile_10 = np.transpose(pred_10[-fcst_len: ], (1, 0, 2))
        quantile_90 = np.transpose(pred_90[-fcst_len: ], (1, 0, 2))
        
        batch_size, length, _ = prediction.shape
        city_id_col, stat_date_col = [], []
        
        for city_id, start_date in zip(city_ids, start_dates):
            city_id_col += [city_id for _ in range(length)]
            
            date_range = list(pd.date_range(config['start_test'], 
                                            config['start_test'] + datetime.timedelta(days=length),
                                            closed='left'))
            date_range = [date.strftime('%Y-%m-%d') for date in date_range]
            stat_date_col += date_range
        
        # convert prediction for batch to DataFrame
        result_dict = {'city_id': city_id_col, 
                       'stat_date': stat_date_col}  # (nb * length, )
        for metric in config['metrics']:
            ind = config['metric_index'][metric]
            result_dict[metric] = np.reshape(prediction[:, :, ind], (-1, ))
            result_dict[metric + '_10'] = np.reshape(quantile_10[:, :, ind], (-1, ))
            result_dict[metric + '_90'] = np.reshape(quantile_90[:, :, ind], (-1, ))

        result_dict = pd.DataFrame(result_dict)
        preds = pd.concat([preds, result_dict])
    
    city_list = sorted(list(set(preds.city_id)))
    for city_id in city_list:
        for metric in config['metrics']:
            metric_scale = scale.loc[scale.city_id == city_id, metric + '_scale'].values[0]
            preds.loc[preds.city_id == city_id, metric] *= metric_scale
            preds.loc[preds.city_id == city_id, metric + '_10'] *= metric_scale
            preds.loc[preds.city_id == city_id, metric + '_90'] *= metric_scale
    
    # evaluate the results when all test data is available
    if max(preds.stat_date) <= max(raw_data.stat_date):
        mape, mape_df = evaluate(preds, raw_data, config)
        q50 = normalized_quantile_loss(preds, raw_data, config)
        q10 = normalized_quantile_loss(preds, raw_data, config, Q=10)
        q90 = normalized_quantile_loss(preds, raw_data, config, Q=90)
    
        for ind, metric in enumerate(config['metrics']):
            glog.info(' Test %25s q10 / q50 / q90 losses = %.4f / %.4f / %.4f'
                          % (metric, 
                             q10[ind], 
                             q50[ind],
                             q90[ind]))

        for metric in config['metrics']:
            glog.info(' Test %25s mape = %.4f'
                          % (metric, mape[config['metric_index'][metric]]))
        glog.info('\n%s' % mape_df.loc[mape_df.city_id <= 150, config['metrics']].describe())
                        
    return preds

    
def valid(model, data_set, config):
    extr_len = config['extra_length']
    preds, labels, loss = np.array([]), np.array([]), []
    for iter in range(data_set.batch_num):
        (input, label, city_onehot, mask, time), info = data_set.iterate(return_info=True)
        
        pred, _, _, l = model.infer(input, label, city_onehot, mask, time, return_loss=True)
        loss.append(l)
        
        city_ids, start_dates = info[1], info[2]
        label = label.cpu().data.numpy()
        
        pred  = pred[-config['forecast_length']: , :, :]
        label = label[-config['forecast_length']: , :, :]
        
        preds = np.concatenate((preds, pred), axis=1) if preds.shape != (0, ) else pred
        labels = np.concatenate((labels, label), axis=1) if labels.shape != (0, ) else label
    
    abs_error = np.abs(preds / (labels + 1e-6) - 1.)  # (length, sample_num, num_out)
    abs_error = np.median(np.mean(abs_error, axis=0), axis=0)
    
    glog.info('Valid abs_loss = %.4f' % np.mean(loss)) 
    for metric in config['metrics']:
        glog.info('Valid %25s median mape = %.4f' 
                     % (metric, abs_error[config['metric_index'][metric]]))
        
    return np.mean(loss)


def run(model, train_set, valid_set, test_set, scale, raw_data, config): 
    loss_base, iter_temp = 100., 0
    
    for epoch in range(config['max_epoch']):
        losses_abs, losses_sum, losses_reg = [], [], []
        
        for iter in range(train_set.batch_num):
            inputs = train_set.iterate()
            errors = model.train(*inputs)
            model.scheduler.step()
            
            losses_abs.append(errors[0])
            losses_sum.append(errors[1])
            losses_reg.append(errors[2])
            
            if (iter + 1) % config['iter_disp'] == 0:
                # lr = model.scheduler.get_lr()[0]
                lr = model.scheduler._rate
                glog.info('Epoch %d, Iteration %d, lr = %.2e, abs_loss = %.4f, sum_loss = %.4f, reg_loss = %.4f' 
                             % (epoch, iter + 1, lr, np.mean(losses_abs), np.mean(losses_sum), np.mean(losses_reg)))
                losses_abs, losses_sum, losses_reg = [], [], []
                
            if (iter + 1) % config['iter_save'] == 0:
                loss_temp = valid(model, valid_set, config)
                predictions = test(model, test_set, scale, raw_data, config)
                      
                save_file = os.path.join(config['snap_path'], 'snapshot', 'epoch_%02d_iter_%05d' 
                                         % (epoch, iter + 1))
                glog.info('Snap to path: %s' % save_file)
                torch.save(model, save_file)
                
                result_file = os.path.join(config['snap_path'], 'result', 'epoch_%02d_iter_%05d.csv' 
                                           % (epoch, iter + 1))
                predictions.to_csv(result_file, index=False)      
                
                
if __name__ == '__main__':
    glog.info('beginning ...')
    save_backup(os.getcwd(), config)
    glog.info('to save snapshots and results to path: %s' % config['snap_path'])

    glog.info('building database ...')
    raw_data, outputs, augmented_scale, scale = load_database(config)
    save_database(os.getcwd(), config)
    
    train_set = Reader(*outputs, augmented_scale, config, 'train')
    valid_set = Reader(*outputs, augmented_scale, config, 'valid')
    test_set = Reader(*outputs, augmented_scale, config, 'test')
    
    model = Model(config)
    
    run(model,
        train_set, 
        valid_set,
        test_set,
        scale,
        raw_data,
        config)