"""
Author: Zaw
Predicting the LogP of molecules by using GB
"""

import numpy as np
import pandas as pd
import math
import time
import os
import sys
import warnings
warnings.simplefilter(action='ignore')
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, 
from sklearn.model_selection import train_test_split
from hyperopt import hp
start = time.time()
import pickle

# Rescale the feature
def scale(col):
    return (col - np.min(col))/(np.max(col)-np.min(col))

## LC informations
# Import LC informations
path = os.getcwd()
data_path = path+"/data/extract_data.csv"
col_list=['LogP','Exp_RT']
lc_df = pd.read_csv(data_path,usecols=col_list)

# Remove non_retained molecules
index=lc_df[lc_df['Exp_RT'] < 180].index
lc_df.drop(lc_df[lc_df['Exp_RT'] < 180].index,inplace=True)

# Import descriptor file
path = os.getcwd()
data_path = path+"/data/descriptors.csv"
col_list = ['PEOE_VSA7','VSA_EState6','PEOE_VSA6','TPSA','MolMR','BCUT2D_LOGPLOW','EState_VSA8','NumAromaticCarbocycles','SlogP_VSA2']#'SlogP_VSA2','VSA_EState8','VSA_EState10','BCUT2D_LOGPHI','SlogP_VSA10','SlogP_VSA5','Kappa3','fr_Ar_OH','EState_VSA1','fr_halogen']
#col_list = ['LabuteASA','PEOE_VSA1','VSA_EState2','PEOE_VSA14','Kappa3','Chi1','TPSA','NumHAcceptors','RingCount','PEOE_VSA7']
des_df = pd.read_csv(data_path,usecols=col_list)

# Remove non_retained molecules
des_df  = des_df.drop(des_df.index[[index]])


# The dataset without lc informations
#load the model without lc
path = os.getcwd()
dirname = os.path.dirname(path)
file_path = dirname+"/logP/hyperparameters/xgb_lc.pkl"
best_parameters_with_lc = pickle.load(open(file_path,'rb'))

space = {'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
          'gamma': hp.uniform('gamma', 0, 0.2),
          'min_child_weight': hp.choice('min_child_weight', range(1, 6)),
          'subsample': hp.uniform('subsample', 0.7, 1.0),
          'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
          'max_depth': hp.choice('max_depth', range(3, 10)),
          'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300, 400, 500, 1000])}
min_child_weight_ls = range(1, 6)
max_depth_ls = range(3, 10)
n_estimators_ls = [100, 200, 300, 400, 500, 1000, 1500, 2000]

# The dataset without lc informations
#load the model without lc



# #Single random run without RT
# def run_best_model(arg_1,arg_2):
#     data_set_1 = arg_1
#     data_set_2 = arg_2
#     des_without_lc = pd.concat([data_set_1,data_set_2['LogP']],axis=1)
#     des_without_lc_feat_corr = des_without_lc.columns[des_without_lc.corrwith(des_without_lc['LogP']) >=0.90][:-1]
#     des_without_lc = des_without_lc.drop(columns=des_without_lc_feat_corr)

#     # Filling the nan with mean values in des_with_lc
#     for col in des_without_lc:
#         des_without_lc[col].fillna(des_without_lc[col].mean(),inplace=True)

#     # Remove columns with zero vlues
#     des_without_lc = des_without_lc.loc[:,(des_without_lc**2).sum() != 0]
#     data = des_without_lc.drop(['LogP'],axis=1)

#     # Remove features with low Variance(threshold <= 0.05)
#     data_var = data.var()
#     del_feat = list(data_var[data_var <= 0.05].index)
#     data.drop(columns=del_feat, inplace=True)

#     # Remove features with correlation(threshold >0.95)
#     corr_matrix = data.corr().abs()
#     mask = np.triu(np.ones_like(corr_matrix,dtype=bool))
#     tri_df = corr_matrix.mask(mask)
#     to_drop =  [ c for c in tri_df.columns if any(tri_df[c] > 0.95)]
#     data = data.drop(to_drop,axis=1)

#     # Scale the features
#     cols = list(data)
#     data[cols] = data[cols].apply(scale,axis=0)
#     data = pd.concat([data,des_without_lc['LogP']],axis=1)

#     # single run
#     # data set preparation

#     train , rest = train_test_split(data,train_size = 0.8,shuffle=True)
#     validate , test = train_test_split(rest, train_size = 0.5, shuffle=True)

#     # training set
#     data_tra_x = train.drop(['LogP'],axis=1)
#     data_tra_y = train['LogP']

#     # validation set
#     data_val_x = validate.drop(['LogP'],axis=1)
#     data_val_y = validate['LogP']

#     # test set
#     data_tes_x = test.drop(['LogP'],axis=1)
#     data_tes_y = test['LogP']

#     best_model = XGBRegressor( n_estimators=n_estimators_ls[best_parameters_without_lc['n_estimators']], 
#                                 max_depth=max_depth_ls[best_parameters_without_lc['max_depth']],
#                                 min_child_weight=min_child_weight_ls[best_parameters_without_lc['min_child_weight']],
#                                 learning_rate=best_parameters_without_lc['learning_rate'],
#                                 gamma=best_parameters_without_lc['gamma'],
#                                 subsample=best_parameters_without_lc['subsample'],
#                                 colsample_bytree=best_parameters_without_lc['colsample_bytree'],
#                                 n_jobs=6, random_state=1, seed=1)

#     best_model.fit(data_tra_x,data_tra_y)

#     # training error
#     tra_pred = best_model.predict(data_tra_x)
#     tra_results = ['tra',mean_squared_error(data_tra_y,tra_pred),mean_squared_error(data_tra_y,tra_pred,squared=False),
#                 mean_absolute_error(data_tra_y,tra_pred),
#                 r2_score(data_tra_y,tra_pred)]

#     # validation error
#     val_pred = best_model.predict(data_val_x)
#     val_results = ['val',mean_squared_error(data_val_y,val_pred),mean_squared_error(data_val_y,val_pred,squared=False),
#                 mean_absolute_error(data_val_y,val_pred),
#                 r2_score(data_val_y,val_pred)]
#     # testing error
#     tes_pred = best_model.predict(data_tes_x)
#     tes_results = ['tes',mean_squared_error(data_tes_y,tes_pred),mean_squared_error(data_tes_y,tes_pred,squared=False),
#                 mean_absolute_error(data_tes_y,tes_pred),
#                 r2_score(data_tes_y,tes_pred)]


#     all_set = tra_results,val_results,tes_results


#     all_set_df = pd.DataFrame(all_set,columns=['set','mse','rmse','mae','r2'])

#     all_set_df.to_excel(dirname+"/p_chem/results/xgb_single_no_rt.xlsx")

#     print('\nSingle random run without LC information is done!!!')

# run_best_model(des_df,lc_df)

# print('\n50 repetition run without LC information is started...................................')
# #### run 50 repetitions without RT
# splits = 50
# def run_best_model(arg_1,arg_2):
#     data_set_1 = arg_1
#     data_set_2 = arg_2
#     des_without_lc = pd.concat([data_set_1,data_set_2['LogP']],axis=1)
#     des_without_lc_feat_corr = des_without_lc.columns[des_without_lc.corrwith(des_without_lc['LogP']) >=0.90][:-1]
#     des_without_lc = des_without_lc.drop(columns=des_without_lc_feat_corr)

#     # Filling the nan with mean values in des_with_lc
#     for col in des_without_lc:
#         des_without_lc[col].fillna(des_without_lc[col].mean(),inplace=True)

#     # Remove columns with zero vlues
#     des_without_lc = des_without_lc.loc[:,(des_without_lc**2).sum() != 0]
#     data = des_without_lc.drop(['LogP'],axis=1)

#     # Remove features with low Variance(threshold <= 0.05)
#     data_var = data.var()
#     del_feat = list(data_var[data_var <= 0.05].index)
#     data.drop(columns=del_feat, inplace=True)

#     # Remove features with correlation(threshold >0.95)
#     corr_matrix = data.corr().abs()
#     mask = np.triu(np.ones_like(corr_matrix,dtype=bool))
#     tri_df = corr_matrix.mask(mask)
#     to_drop =  [ c for c in tri_df.columns if any(tri_df[c] > 0.95)]
#     data = data.drop(to_drop,axis=1)

#     # Scale the features
#     cols = list(data)
#     data[cols] = data[cols].apply(scale,axis=0)
#     data = pd.concat([data,des_without_lc['LogP']],axis=1)

#     # run 50 repetitions
#     all_set = []
#     for split in range(1,splits+1):
#         seed = split

#         # data set preparation

#         train , rest = train_test_split(data,train_size = 0.8,shuffle=True,random_state = seed)
#         validate , test = train_test_split(rest, train_size = 0.5, shuffle=True,random_state = seed)

#         # training set
#         data_tra_x = train.drop(['LogP'],axis=1)
#         data_tra_y = train['LogP']

#         # validation set
#         data_val_x = validate.drop(['LogP'],axis=1)
#         data_val_y = validate['LogP']

#         # test set
#         data_tes_x = test.drop(['LogP'],axis=1)
#         data_tes_y = test['LogP']

#         best_model = XGBRegressor( n_estimators=n_estimators_ls[best_parameters_without_lc['n_estimators']], 
#                                 max_depth=max_depth_ls[best_parameters_without_lc['max_depth']],
#                                 min_child_weight=min_child_weight_ls[best_parameters_without_lc['min_child_weight']],
#                                 learning_rate=best_parameters_without_lc['learning_rate'],
#                                 gamma=best_parameters_without_lc['gamma'],
#                                 subsample=best_parameters_without_lc['subsample'],
#                                 colsample_bytree=best_parameters_without_lc['colsample_bytree'],
#                                 n_jobs=6, random_state=1, seed=1)

#         best_model.fit(data_tra_x,data_tra_y)

#         # training error
#         tra_pred = best_model.predict(data_tra_x)
#         tra_results = [split,'tra',mean_squared_error(data_tra_y,tra_pred),
#                     mean_squared_error(data_tra_y,tra_pred,squared=False),
#                     mean_absolute_error(data_tra_y,tra_pred),
#                     r2_score(data_tra_y,tra_pred)]

#         # validation error
#         val_pred = best_model.predict(data_val_x)
#         val_results = [' ','val',mean_squared_error(data_val_y,val_pred),
#                     mean_squared_error(data_val_y,val_pred,squared=False),
#                     mean_absolute_error(data_val_y,val_pred),
#                     r2_score(data_val_y,val_pred)]
#         # testing error
#         tes_pred = best_model.predict(data_tes_x)
#         tes_results = [' ','tes',mean_squared_error(data_tes_y,tes_pred),
#                     mean_squared_error(data_tes_y,tes_pred,squared=False),
#                     mean_absolute_error(data_tes_y,tes_pred),
#                     r2_score(data_tes_y,tes_pred)]


#         all_set += tra_results,val_results,tes_results


#     all_set_df = pd.DataFrame(all_set,columns=['split','set','mse','rmse','mae','r2'])

#     data_res =  [   
#                     ['MSE',str(format(all_set_df[all_set_df['set'] == 'tra']['mse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tra']['mse'].std(),".3f")),
#                            str(format(all_set_df[all_set_df['set'] == 'val']['mse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'val']['mse'].std(),".3f")),
#                            str(format(all_set_df[all_set_df['set'] == 'tes']['mse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tes']['mse'].std(),".3f"))],
                    
#                     ['RMSE',str(format(all_set_df[all_set_df['set'] == 'tra']['rmse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tra']['rmse'].std(),".3f")),
#                             str(format(all_set_df[all_set_df['set'] == 'val']['rmse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'val']['rmse'].std(),".3f")),
#                             str(format(all_set_df[all_set_df['set'] == 'tes']['rmse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tes']['rmse'].std(),".3f"))],

#                     ['MAE',str(format(all_set_df[all_set_df['set'] == 'tra']['mae'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tra']['mae'].std(),".3f")),
#                            str(format(all_set_df[all_set_df['set'] == 'val']['mae'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'val']['mae'].std(),".3f")),
#                            str(format(all_set_df[all_set_df['set'] == 'tes']['mae'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tes']['mae'].std(),".3f"))],

#                     ['R2',str(format(all_set_df[all_set_df['set'] == 'tra']['r2'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tra']['r2'].std(),".3f")),
#                           str(format(all_set_df[all_set_df['set'] == 'val']['r2'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'val']['r2'].std(),".3f")),
#                           str(format(all_set_df[all_set_df['set'] == 'tes']['r2'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tes']['r2'].std(),".3f"))]
#                 ] 

#     final = pd.DataFrame(data_res,columns = ['', 'Training',' Validation', 'Testing'])
#     final.to_excel(dirname+"/p_chem/results/xgb_50_no_rt.xlsx")

# run_best_model(des_df,lc_df)
# print('50 repetition run without LC information is done!!!!')



# # The dataset with lc informations
# #load the model with lc
# path = os.getcwd()
# dirname = os.path.dirname(path)
# file_path = dirname+"/p_chem/hyperparameters/xgb_lc.pkl"
# best_parameters_with_lc = pickle.load(open(file_path,'rb'))


#Single random run without RT
def run_best_model(arg_1,arg_2):
    data_set_1 = arg_1
    data_set_2 = arg_2
    des_without_lc = pd.concat([data_set_1,data_set_2['LogP']],axis=1)
    des_without_lc_feat_corr = des_without_lc.columns[des_without_lc.corrwith(des_without_lc['LogP']) >=0.90][:-1]
    des_without_lc = des_without_lc.drop(columns=des_without_lc_feat_corr)

    # Filling the nan with mean values in des_with_lc
    for col in des_without_lc:
        des_without_lc[col].fillna(des_without_lc[col].mean(),inplace=True)

    # Remove columns with zero vlues
    des_without_lc = des_without_lc.loc[:,(des_without_lc**2).sum() != 0]
    data = des_without_lc.drop(['LogP'],axis=1)

    # Remove features with low Variance(threshold <= 0.05)
    data_var = data.var()
    del_feat = list(data_var[data_var <= 0.05].index)
    data.drop(columns=del_feat, inplace=True)

    # Remove features with correlation(threshold >0.95)
    corr_matrix = data.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix,dtype=bool))
    tri_df = corr_matrix.mask(mask)
    to_drop =  [ c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    data = data.drop(to_drop,axis=1)

    # Scale the features
    cols = list(data)
    data[cols] = data[cols].apply(scale,axis=0)
    data = pd.concat([data,des_without_lc['LogP']],axis=1)

    # single run
    # data set preparation

    train , rest = train_test_split(data,train_size = 0.8,shuffle=True)
    validate , test = train_test_split(rest, train_size = 0.5, shuffle=True)

    # training set
    data_tra_x = train.drop(['LogP'],axis=1)
    data_tra_y = train['LogP']

    # validation set
    data_val_x = validate.drop(['LogP'],axis=1)
    data_val_y = validate['LogP']

    # test set
    data_tes_x = test.drop(['LogP'],axis=1)
    data_tes_y = test['LogP']

    best_model = XGBRegressor( n_estimators=n_estimators_ls[best_parameters_with_lc['n_estimators']], 
                                max_depth=max_depth_ls[best_parameters_with_lc['max_depth']],
                                min_child_weight=min_child_weight_ls[best_parameters_with_lc['min_child_weight']],
                                learning_rate=best_parameters_with_lc['learning_rate'],
                                gamma=best_parameters_with_lc['gamma'],
                                subsample=best_parameters_with_lc['subsample'],
                                colsample_bytree=best_parameters_with_lc['colsample_bytree'],
                                n_jobs=6, random_state=1, seed=1)

    best_model.fit(data_tra_x,data_tra_y)

    # training error
    tra_pred = best_model.predict(data_tra_x)
    tra_results = ['tra',mean_squared_error(data_tra_y,tra_pred),mean_squared_error(data_tra_y,tra_pred,squared=False),
                mean_absolute_error(data_tra_y,tra_pred),
                r2_score(data_tra_y,tra_pred)]

    # validation error
    val_pred = best_model.predict(data_val_x)
    val_results = ['val',mean_squared_error(data_val_y,val_pred),mean_squared_error(data_val_y,val_pred,squared=False),
                mean_absolute_error(data_val_y,val_pred),
                r2_score(data_val_y,val_pred)]
    # testing error
    tes_pred = best_model.predict(data_tes_x)
    tes_results = ['tes',mean_squared_error(data_tes_y,tes_pred),mean_squared_error(data_tes_y,tes_pred,squared=False),
                mean_absolute_error(data_tes_y,tes_pred),
                r2_score(data_tes_y,tes_pred)]


    all_set = tra_results,val_results,tes_results


    all_set_df = pd.DataFrame(all_set,columns=['set','mse','rmse','mae','r2'])
    all_set_df.to_excel(dirname+"/logP/results/xgb_single_rt.xlsx")
    print('\nSingle random run with LC information is done.')


run_best_model(des_df,lc_df)


# print('\n50 repetition run with LC information is started...........................')
#### run 50 repetitions with RT
splits = 50
def run_best_model(arg_1,arg_2):
    data_set_1 = arg_1
    data_set_2 = arg_2
    des_with_lc = pd.concat([data_set_1,data_set_2],axis=1)
    des_with_lc_feat_corr = des_with_lc.columns[des_with_lc.corrwith(des_with_lc['LogP']) >=0.90][:-1]
    des_with_lc = des_with_lc.drop(columns=des_with_lc_feat_corr)

    # Filling the nan with mean values in des_with_lc
    for col in des_with_lc:
        des_with_lc[col].fillna(des_with_lc[col].mean(),inplace=True)

    # Remove columns with zero vlues
    des_with_lc = des_with_lc.loc[:,(des_with_lc**2).sum() != 0]
    data = des_with_lc.drop(['LogP'],axis=1)

    # Remove features with low Variance(threshold<=0.05)
    data_var = data.var()
    del_feat = list(data_var[data_var <= 0.05].index)
    data.drop(columns=del_feat, inplace=True)

    # Remove features with correlation(threshold > 0.95)
    corr_matrix = data.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix,dtype=bool))
    tri_df = corr_matrix.mask(mask)
    to_drop =  [ c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    data = data.drop(to_drop,axis=1)

    # Scale the features
    cols = list(data)
    data[cols] = data[cols].apply(scale,axis=0)
    data = pd.concat([data,des_with_lc['LogP']],axis=1)

    # run 50 repetitions
    all_set = []
    for split in range(1,splits+1):
        seed = split

        # data set preparation
        train , rest = train_test_split(data,train_size = 0.8,shuffle=True,random_state = seed)
        validate , test = train_test_split(rest, train_size = 0.5, shuffle=True,random_state = seed)

        # training set
        data_tra_x = train.drop(['LogP'],axis=1)
        data_tra_y = train['LogP']

        # validation set
        data_val_x = validate.drop(['LogP'],axis=1)
        data_val_y = validate['LogP']

        # test set
        data_tes_x = test.drop(['LogP'],axis=1)
        data_tes_y = test['LogP']

        best_model = XGBRegressor()
        # ( n_estimators=n_estimators_ls[best_parameters_with_lc['n_estimators']], 
        #                         max_depth=max_depth_ls[best_parameters_with_lc['max_depth']],
        #                         min_child_weight=min_child_weight_ls[best_parameters_with_lc['min_child_weight']],
        #                         learning_rate=best_parameters_with_lc['learning_rate'],
        #                         gamma=best_parameters_with_lc['gamma'],
        #                         subsample=best_parameters_with_lc['subsample'],
        #                         colsample_bytree=best_parameters_with_lc['colsample_bytree'],
        #                         n_jobs=6, random_state=1, seed=1)

        best_model.fit(data_tra_x,data_tra_y)

        # training error
        tra_pred = best_model.predict(data_tra_x)
        tra_results = [split,'tra',mean_squared_error(data_tra_y,tra_pred),
                    mean_squared_error(data_tra_y,tra_pred,squared=False),
                    mean_absolute_error(data_tra_y,tra_pred),
                    r2_score(data_tra_y,tra_pred)]

        # validation error
        val_pred = best_model.predict(data_val_x)
        val_results = [' ','val',mean_squared_error(data_val_y,val_pred),
                    mean_squared_error(data_val_y,val_pred,squared=False),
                    mean_absolute_error(data_val_y,val_pred),
                    r2_score(data_val_y,val_pred)]
        # testing error
        tes_pred = best_model.predict(data_tes_x)
        tes_results = [' ','tes',mean_squared_error(data_tes_y,tes_pred),
                    mean_squared_error(data_tes_y,tes_pred,squared=False),
                    mean_absolute_error(data_tes_y,tes_pred),
                    r2_score(data_tes_y,tes_pred)]


        all_set += tra_results,val_results,tes_results


    all_set_df = pd.DataFrame(all_set,columns=['split','set','mse','rmse','mae','r2'])
    data_res =  [   
                    ['MSE',str(format(all_set_df[all_set_df['set'] == 'tra']['mse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tra']['mse'].std(),".3f")),
                           str(format(all_set_df[all_set_df['set'] == 'val']['mse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'val']['mse'].std(),".3f")),
                           str(format(all_set_df[all_set_df['set'] == 'tes']['mse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tes']['mse'].std(),".3f"))],
                    
                    ['RMSE',str(format(all_set_df[all_set_df['set'] == 'tra']['rmse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tra']['rmse'].std(),".3f")),
                            str(format(all_set_df[all_set_df['set'] == 'val']['rmse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'val']['rmse'].std(),".3f")),
                            str(format(all_set_df[all_set_df['set'] == 'tes']['rmse'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tes']['rmse'].std(),".3f"))],

                    ['MAE',str(format(all_set_df[all_set_df['set'] == 'tra']['mae'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tra']['mae'].std(),".3f")),
                           str(format(all_set_df[all_set_df['set'] == 'val']['mae'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'val']['mae'].std(),".3f")),
                           str(format(all_set_df[all_set_df['set'] == 'tes']['mae'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tes']['mae'].std(),".3f"))],

                    ['R2',str(format(all_set_df[all_set_df['set'] == 'tra']['r2'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tra']['r2'].std(),".3f")),
                          str(format(all_set_df[all_set_df['set'] == 'val']['r2'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'val']['r2'].std(),".3f")),
                          str(format(all_set_df[all_set_df['set'] == 'tes']['r2'].mean(),".3f"))+'+/-'+str(format(all_set_df[all_set_df['set'] == 'tes']['r2'].std(),".3f"))]
                ] 

    final = pd.DataFrame(data_res,columns = ['', 'Training',' Validation', 'Testing'])
    final.to_excel(dirname+"/logP/results/top_10_xgb_50_rt.xlsx")
   
run_best_model(des_df,lc_df)
print('50 repetition run with LC information is done!!!!')


end = time.time()  # get the end time
print('\nThe total elapsed time is:', (end - start), 'S')
