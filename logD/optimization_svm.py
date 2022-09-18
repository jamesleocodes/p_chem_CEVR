# Hyperparameter optimization without LC for SVM


import numpy as np
import pandas as pd
import math
import time
import os
import sys
import warnings
warnings.simplefilter(action='ignore')
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from hyperopt import fmin,tpe,hp,STATUS_OK, Trials
import pickle

start = time.time()
# Rescale the feature
def scale(col):
    return (col - np.min(col))/(np.max(col)-np.min(col))


#mean percent error
## LC informations
# Import LC informations
path = os.getcwd()
dirname = os.path.dirname(path)
data_path = dirname+"/p_chem_logd/data/extract_data.csv"
col_list=['LogD','Exp_RT']
lc_df = pd.read_csv(data_path,usecols=col_list)

# Remove non_retained molecules
index=lc_df[lc_df['Exp_RT'] < 180].index
lc_df.drop(lc_df[lc_df['Exp_RT'] < 180].index,inplace=True)

# Import descriptor file
path = os.getcwd()
dirname = os.path.dirname(path)
data_path = dirname+"/p_chem_logd/data/descriptors.csv"
des_df = pd.read_csv(data_path,index_col=0)

# Remove non_retained molecules
des_df  = des_df.drop(des_df.index[[index]])

space = {'C':hp.uniform('C',0.1,100)}

####################
# The dataset without lc informations
# Hyper-parammeter tuning without RT
def hyper_optimize(arg_1,arg_2):
    data_set_1 = arg_1
    data_set_2 = arg_2
    des_without_lc = pd.concat([data_set_1,data_set_2['LogD']],axis=1)
    des_without_lc_feat_corr = des_without_lc.columns[des_without_lc.corrwith(des_without_lc['LogD']) >=0.90][:-1]
    des_without_lc = des_without_lc.drop(columns=des_without_lc_feat_corr)

    # Filling the nan with mean values in des_with_lc
    for col in des_without_lc:
        des_without_lc[col].fillna(des_without_lc[col].mean(),inplace=True)

    # Remove columns with zero vlues
    des_without_lc = des_without_lc.loc[:,(des_without_lc**2).sum() != 0]
    data = des_without_lc.drop(['LogD'],axis=1)

    # Remove features with low Variance(threshold<=0.05)
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
    data = pd.concat([data,des_without_lc['LogD']],axis=1)

    # data set preparation
    train , rest = train_test_split(data,train_size = 0.8,shuffle=True,random_state = 3 )
    validate , test = train_test_split(rest, train_size = 0.5, shuffle=True,random_state = 3)

    # training set
    data_tra_x = train.drop(['LogD'],axis=1)
    data_tra_y = train['LogD']

    # validation set
    data_val_x = validate.drop(['LogD'],axis=1)
    data_val_y = validate['LogD']

    # test set
    data_tes_x = test.drop(['LogD'],axis=1)
    data_tes_y = test['LogD']
    print('\nThe optimization started.............')
    # Start hyper parameter optimization

    # Define a objective function
    def hyperparameter_tuning(params):
        model = svm.SVR(**params,kernel='linear',cache_size=2000,max_iter=10000)
        model.fit(data_tra_x,data_tra_y)
        val_preds = model.predict(data_val_x)
        loss = mean_squared_error(data_val_y,val_preds,squared=False)
        # X = data.drop(['LogD'],axis=1)
        # y = data['LogD']
        # acc = cross_val_score(model,X,y,scoring ='neg_mean_squared_error').mean()
        # return {'loss': np.sqrt(-acc),'status':STATUS_OK}
        return {'loss': loss,'status':STATUS_OK}

    # Initialize trials object
    trials = Trials()

    # Fine tune the model
    best_results = fmin(
        fn=hyperparameter_tuning,
        space = space,
        algo=tpe.suggest,
        max_evals= 50,
        trials=trials
    )
    return best_results
best_parameters_without_lc = dict(hyper_optimize(des_df,lc_df))

#save the parameter without lc

file_path = dirname+"/p_chem_logd/hyperparameters/svm_no_lc.pkl"
pickle.dump(best_parameters_without_lc, open(file_path,'wb'))
print('The parameters without lc for svm have been saved now.')


# Hyper-parammeter tuning with RT
def hyper_optimize(arg_1,arg_2):
    data_set_1 = arg_1
    data_set_2 = arg_2
    des_with_lc = pd.concat([data_set_1,data_set_2],axis=1)
    des_with_lc_feat_corr = des_with_lc.columns[des_with_lc.corrwith(des_with_lc['LogD']) >=0.90][:-1]
    des_with_lc = des_with_lc.drop(columns=des_with_lc_feat_corr)

    # Filling the nan with mean values in des_with_lc
    for col in des_with_lc:
        des_with_lc[col].fillna(des_with_lc[col].mean(),inplace=True)

    # Remove columns with zero vlues
    des_with_lc = des_with_lc.loc[:,(des_with_lc**2).sum() != 0]
    data = des_with_lc.drop(['LogD'],axis=1)

    # Remove features with low Variance(threshold<=0.05)
    data_var = data.var()
    del_feat = list(data_var[data_var <= 0.05].index)
    data.drop(columns=del_feat, inplace=True)

    # Remove features with correlation(threshold > 0.95) r
    corr_matrix = data.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix,dtype=bool))
    tri_df = corr_matrix.mask(mask)
    to_drop =  [ c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    data = data.drop(to_drop,axis=1)

    # Scale the features
    cols = list(data)
    data[cols] = data[cols].apply(scale,axis=0)
    data = pd.concat([data,des_with_lc['LogD']],axis=1)

    # data set preparation
    train , rest = train_test_split(data,train_size = 0.8,shuffle=True,random_state = 3)
    validate , test = train_test_split(rest, train_size = 0.5, shuffle=True,random_state = 3)

    # training set
    data_tra_x = train.drop(['LogD'],axis=1)
    data_tra_y = train['LogD']

    # validation set
    data_val_x = validate.drop(['LogD'],axis=1)
    data_val_y = validate['LogD']

    # test set
    data_tes_x = test.drop(['LogD'],axis=1)
    data_tes_y = test['LogD']
    print('\nThe optimization started.............')

    # Start hyper parameter optimization
    # Define a objective function
    def hyperparameter_tuning(params):
        model = svm.SVR(**params,kernel='linear',cache_size=2000,max_iter=10000)
        model.fit(data_tra_x,data_tra_y)
        val_preds = model.predict(data_val_x)
        loss = mean_squared_error(data_val_y,val_preds,squared=False)
        return {'loss': loss,'status':STATUS_OK}
    # Initialize trials object
    trials = Trials()

    # Fine tune the model
    best_results = fmin(
        fn=hyperparameter_tuning,
        space = space,
        algo=tpe.suggest,
        max_evals= 50,
        trials=trials
    )
    return best_results
best_parameters_with_lc = dict(hyper_optimize(des_df,lc_df))


#save the parameters with lc
file_path = dirname+"/p_chem_logd/hyperparameters/svm_lc.pkl"
pickle.dump(best_parameters_with_lc, open(file_path,'wb'))
print('The parameters with lc for svm have been saved now.')

end = time.time()  # get the end time
print('\nThe total elapsed time is:', (end - start), 'S')