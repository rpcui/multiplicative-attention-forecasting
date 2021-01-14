import os
import numpy as np
import pandas as pd
import datetime


config = {}

config['mode'] = 'prediction'  # mode for test: evaluation / prediction
config['scale_mode'] = 'linear'

config['metrics'] = ['total_no_call_order_cnt', 
                     'strive_order_cnt', 
                     'total_finish_order_cnt', 
                     'online_time', 
                     'online_payed',
                     'total_gmv']
config['ratios'] = ['finish_ratio', 
                    'strive_ratio',
                    'payed_ratio']

config['metrics_output'] = ['total_no_call_order_cnt',
                            'strive_order_cnt', 
                            'total_finish_order_cnt', 
                            'online_time',
                            'online_payed',
                            'total_gmv']

config['metric_index'] = dict(zip(config['metrics'], 
                                  range(len(config['metrics']))))
config['ratio_index'] = dict(zip(config['ratios'], 
                                 range(len(config['ratios']))))

# to ensure the relations of metrics after normalization
config['norm_constr'] = {'strive_order_cnt': 'total_no_call_order_cnt',
                         'total_finish_order_cnt': 'total_no_call_order_cnt',
                         'online_payed': 'online_time'}

config['pivot_column'] = 'total_no_call_order_cnt'
config['pivot_scales'] = {'total_no_call_order_cnt': 1.0,
                          'strive_order_cnt':        0.87501,
                          'total_finish_order_cnt':  0.77995,
                          'online_time':             0.40696,
                          'online_payed':            0.14371,
                          'total_gmv':               9.47475}


# config['feat_collections'] = ['history', 'weather', 'subsidy', 'holiday', 'temporal']
config['feat_collections'] = ['weather', 'subsidy', 'holiday', 'temporal']
config['feat_to_sum'] = ['history']
config['feat_to_agg'] = ['weather', 'subsidy', 'holiday', 'temporal']
config['reference'] = 'total_finish_order_cnt'


config['weather_columns'] = ['rain_hour_cnt','pm25','intensity', 'temperature', 'wind_speed'] + \
                            ['intensity_%02d' % i for i in range(0, 24)]

config['window_set_for_avg'] = [14, 28]
config['window_set_for_avg_by_weekday'] = [3, 4]
config['num_weekly'] = 3
config['num_yearly'] = 10

config['num_holiday_terms'] = 4

config['stage_range'] = 100
config['min_data_length'] = 365
config['extra_length'] = 13

city_list = set(list(range(1, 366))) - set([31, 204])
config['city_list'] = sorted(list(city_list))
config['num_city'] = 370

config['num_cluster'] = 10
config['num_embed'] = 10

config['num_input'] = 96

config['num_hidden'] = 32
config['num_latent'] = 4

'''
params for Transformer
'''
config['d_model'] = 64
config['nhead'] = 4
config['num_encoder_layers'] = 2
config['num_decoder_layers'] = 2
config['dim_feedforward'] = 32
config['pos_enc_cycles'] = [7.0 / 2, 365.25 / 4]
config['temporal_origin'] = datetime.date(2017, 1, 1)

config['dropout'] = 0.1

config['augment_prob'] = 0.4
config['augment_scale'] = 0.1
config['augment_scale_var'] = 0.01

# begin_test_date = datetime.datetime.now().date()
begin_test_date = datetime.datetime(2019, 9, 10)
# begin_test_date = datetime.datetime(2019, 11, 20)
# begin_test_date = datetime.datetime(2019, 11, 10)

config['forecast_length'] = 28
config['start_train'] = pd.to_datetime('2017-03-01')
config['end_train']   = begin_test_date - datetime.timedelta(days=28)
config['start_valid'] = begin_test_date - datetime.timedelta(days=28)
config['end_valid']   = begin_test_date - datetime.timedelta(days=1)
config['start_test']  = begin_test_date
config['end_test']    = begin_test_date + datetime.timedelta(days=config['forecast_length'] - 1)

config['warm_start_for_train'] = 28
config['warm_start_for_valid'] = 70
config['lw_range'] = 70
config['up_range'] = 140
config['forecast_length_lw_range'] = 21
config['forecast_length_up_range'] = 49

config['batch_size'] = 8
config['learning_rate'] = 5e-5
config['warmup'] = 4000

config['max_epoch'] = 5
config['tolerance'] = 4
config['iter_save'] = 1000
config['iter_disp'] = 100

config['suffix'] = 'decouple_new-scale'
config['snap_path'] = os.path.join('output',
                                   datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + 
                                   '-' + config['suffix'])