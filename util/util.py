import os
import numpy as np
import pandas as pd
import pdb

def save_backup(path, config):
    os.makedirs(os.path.join(path, config['snap_path']))
    os.makedirs(os.path.join(path, config['snap_path'], 'snapshot'))
    os.makedirs(os.path.join(path, config['snap_path'], 'result'))
    
    command = 'cp %s/*.py %s' % (path, config['snap_path'])
    os.system(command)
    
    command = 'cp %s/../utils/*.py %s' % (path, config['snap_path'])
    os.system(command)
    

def save_database(path, config):
    command = 'cp %s/*.pkl %s' % (path, config['snap_path'])
    os.system(command)
    
def normalized_quantile_loss(pred, raw_data, config, Q=50):
    metrics = config['metrics']
    quantile = Q / 100.
    
    gt = raw_data[['city_id', 'stat_date'] + metrics].copy()
    gt = gt.fillna(0)
    merge = pd.merge(pred, gt, on=['city_id', 'stat_date'])
    
    loss = []
    for metric in metrics:
        y = merge[metric + '_y'].values
        if Q == 50:
            y_hat = merge[metric + '_x'].values
        else:
            y_hat = merge['{}_{}'.format(metric, Q)].values
            
        errors = y - y_hat
        weighted_errors = quantile * np.maximum(errors, 0.) \
            + (1. - quantile) * np.maximum(-errors, 0.)

        quantile_loss = weighted_errors.mean()
        normaliser = np.abs(y).mean()
        loss.append(2. * quantile_loss / normaliser)
        
    return loss
    
def evaluate(pred, raw_data, config):
    metrics = config['metrics']
    
    gt = raw_data[['city_id', 'stat_date'] + metrics].copy()
    gt = gt.fillna(0)
    merge = pd.merge(pred, gt, on=['city_id', 'stat_date'], suffixes=('_p', '_g'))
    
    mape_df = merge[['city_id', 'stat_date']]
    for metric in metrics:
        mape_df[metric] = np.abs(merge[metric + '_p'] / (merge[metric + '_g'] + 1e-6) - 1.)
    
    mape = np.median(mape_df[metrics].values, axis=0)
    
    return mape, mape_df
