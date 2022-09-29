"""
Author: Zaw
Predicting the LogD of molecules by using SVM
"""

import numpy as np
import pandas as pd
import time
import os
import warnings
warnings.simplefilter(action='ignore')
from sklearn import svm
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import pickle
start = time.time()


# Rescale the feature
def scale(col):
    return (col - np.min(col))/(np.max(col)-np.min(col))

# Import LC informations
path = os.getcwd()
dirname = os.path.dirname(path)
data_path = dirname+"/p_chem/data/extract_data.csv"
col_list=['LogD','Exp_RT']
lc_df = pd.read_csv(data_path,usecols=col_list)

# Remove non_retained molecules
index=lc_df[lc_df['Exp_RT'] < 180].index
lc_df.drop(lc_df[lc_df['Exp_RT'] < 180].index,inplace=True)

# Import descriptor file
path = os.getcwd()
dirname = os.path.dirname(path)
data_path = dirname+"/p_chem/data/descriptors.csv"
des_df = pd.read_csv(data_path,index_col=0)

# Remove non_retained molecules
des_df  = des_df.drop(des_df.index[[index]])

#load the model without lc
path = os.getcwd()
dirname = os.path.dirname(path)
file_path = dirname+"/p_chem/hyperparameters/svm_no_lc.pkl"
best_parameters_without_lc = pickle.load(open(file_path,'rb'))

####
# Single random run without RT
def run_best_model(arg_1,arg_2):
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
    data = pd.concat([data,des_without_lc['LogD']],axis=1)

    # single run


    # data set preparation

    train , rest = train_test_split(data,train_size = 0.8,shuffle=True)
    validate , test = train_test_split(rest, train_size = 0.5, shuffle=True)

    # training set
    data_tra_x = train.drop(['LogD'],axis=1)
    data_tra_y = train['LogD']

    # validation set
    data_val_x = validate.drop(['LogD'],axis=1)
    data_val_y = validate['LogD']

    # test set
    data_tes_x = test.drop(['LogD'],axis=1)
    data_tes_y = test['LogD']

    best_model = svm.SVR(kernel='linear',cache_size=2000,max_iter=10000, C = best_parameters_without_lc['C'])

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
    all_set_df.to_excel(dirname+"/p_chem/results/svm_single_no_rt.xlsx")

    print('\nSingle random run without LC information is done.')
run_best_model(des_df,lc_df)


print('\n50 repetition run without LC information is started...................................')
#### run 50 repetitions without RT
splits = 50
def run_best_model(arg_1,arg_2):
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
    data = pd.concat([data,des_without_lc['LogD']],axis=1)

    # run 50 repetitions
    all_set = []
    for split in range(1,splits+1):
        seed = split

        # data set preparation

        train , rest = train_test_split(data,train_size = 0.8,shuffle=True,random_state = seed)
        validate , test = train_test_split(rest, train_size = 0.5, shuffle=True,random_state = seed)

        # training set
        data_tra_x = train.drop(['LogD'],axis=1)
        data_tra_y = train['LogD']

        # validation set
        data_val_x = validate.drop(['LogD'],axis=1)
        data_val_y = validate['LogD']

        # test set
        data_tes_x = test.drop(['LogD'],axis=1)
        data_tes_y = test['LogD']

        best_model = svm.SVR(kernel='linear',cache_size=2000,max_iter=10000, C = best_parameters_without_lc['C'])

        best_model.fit(data_tra_x,data_tra_y)

        # training error
        tra_pred = best_model.predict(data_tra_x)
        tra_results = [split,'tra',mean_squared_error(data_tra_y,tra_pred),mean_squared_error(data_tra_y,tra_pred,squared=False),
                    mean_absolute_error(data_tra_y,tra_pred),
                    r2_score(data_tra_y,tra_pred)]

        # validation error
        val_pred = best_model.predict(data_val_x)
        val_results = [' ','val',mean_squared_error(data_val_y,val_pred),mean_squared_error(data_val_y,val_pred,squared=False),
                    mean_absolute_error(data_val_y,val_pred),
                    r2_score(data_val_y,val_pred)]
        # testing error
        tes_pred = best_model.predict(data_tes_x)
        tes_results = [' ','tes',mean_squared_error(data_tes_y,tes_pred),mean_squared_error(data_tes_y,tes_pred,squared=False),
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
    final.to_excel(dirname+"/p_chem/results/svm_50_no_rt.xlsx")
run_best_model(des_df,lc_df)
print('50 repetition run without LC information is done!!!!')


#load the model with lc
path = os.getcwd()
dirname = os.path.dirname(path)
file_path = dirname+"/p_chem/hyperparameters/svm_lc.pkl"
best_parameters_with_lc = pickle.load(open(file_path,'rb'))

#Single random run with RT
def run_best_model(arg_1,arg_2):
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

    # Remove features with correlation(threshold > 0.95)
    corr_matrix = data.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix,dtype=bool))
    tri_df = corr_matrix.mask(mask)
    to_drop =  [ c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    data = data.drop(to_drop,axis=1)

    # Scale the features
    cols = list(data)
    data[cols] = data[cols].apply(scale,axis=0)
    data = pd.concat([data,des_with_lc['LogD']],axis=1)

    # Single random run

    # data set preparation
    train , rest = train_test_split(data,train_size = 0.8,shuffle=True)
    validate , test = train_test_split(rest, train_size = 0.5, shuffle=True)

    # training set
    data_tra_x = train.drop(['LogD'],axis=1)
    data_tra_y = train['LogD']

    # validation set
    data_val_x = validate.drop(['LogD'],axis=1)
    data_val_y = validate['LogD']

    # test set
    data_tes_x = test.drop(['LogD'],axis=1)
    data_tes_y = test['LogD']

    best_model = svm.SVR(kernel='linear',cache_size=2000,max_iter=10000, C = best_parameters_with_lc['C'])

    best_model.fit(data_tra_x,data_tra_y)

    # training error
    tra_pred = best_model.predict(data_tra_x)
    tra_results = ['tra',mean_squared_error(data_tra_y,tra_pred),
                mean_squared_error(data_tra_y,tra_pred,squared=False),
                mean_absolute_error(data_tra_y,tra_pred),
                r2_score(data_tra_y,tra_pred)]

    # validation error
    val_pred = best_model.predict(data_val_x)
    val_results = ['val',mean_squared_error(data_val_y,val_pred),
                mean_squared_error(data_val_y,val_pred,squared=False),
                mean_absolute_error(data_val_y,val_pred),
                r2_score(data_val_y,val_pred)]
    # testing error
    tes_pred = best_model.predict(data_tes_x)
    tes_results = ['tes',mean_squared_error(data_tes_y,tes_pred),
                mean_squared_error(data_tes_y,tes_pred,squared=False),
                mean_absolute_error(data_tes_y,tes_pred),
                r2_score(data_tes_y,tes_pred)]


    all_set = tra_results,val_results,tes_results


    all_set_df = pd.DataFrame(all_set,columns=['set','mse','rmse','mae','r2'])
    all_set_df.to_excel(dirname+"/p_chem/results/svm_single_rt.xlsx")
    print('\nSingle random run with LC information is done.')

run_best_model(des_df,lc_df)


print('\n50 repetition run with LC information is started...........................')

# run 50 repetitions with RT
splits = 50
def run_best_model(arg_1,arg_2):
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

    # Remove features with correlation(threshold > 0.95)
    corr_matrix = data.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix,dtype=bool))
    tri_df = corr_matrix.mask(mask)
    to_drop =  [ c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    data = data.drop(to_drop,axis=1)

    # Scale the features
    cols = list(data)
    data[cols] = data[cols].apply(scale,axis=0)
    data = pd.concat([data,des_with_lc['LogD']],axis=1)

    # run 50 repetitions
    all_set = []
    for split in range(1,splits+1):
        seed = split

        # data set preparation
        train , rest = train_test_split(data,train_size = 0.8,shuffle=True,random_state = seed)
        validate , test = train_test_split(rest, train_size = 0.5, shuffle=True,random_state = seed)

        # training set
        data_tra_x = train.drop(['LogD'],axis=1)
        data_tra_y = train['LogD']

        # validation set
        data_val_x = validate.drop(['LogD'],axis=1)
        data_val_y = validate['LogD']

        # test set
        data_tes_x = test.drop(['LogD'],axis=1)
        data_tes_y = test['LogD']

        best_model = svm.SVR(kernel='linear',cache_size=2000,max_iter=10000, C = best_parameters_with_lc['C'])

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
    final.to_excel(dirname+"/p_chem/results/svm_50_rt.xlsx")

    
run_best_model(des_df,lc_df)
print('50 repetition run with LC information is done!!!!')

end = time.time()  # get the end time
print('\nThe total elapsed time is:', (end - start), 'S')
