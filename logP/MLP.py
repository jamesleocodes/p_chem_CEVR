"""
Author: Zaw
Predicting the LogP of molecules by using MLP
"""

import numpy as np
import pandas as pd
import time
import os
import warnings
import pickle
warnings.simplefilter(action='ignore')
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
start = time.time()

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
des_df = pd.read_csv(data_path,index_col=0)

# Remove non_retained molecules
des_df  = des_df.drop(des_df.index[[index]])


# The dataset without lc informations
#load the model without lc
path = os.getcwd()
dirname = os.path.dirname(path)
file_path = dirname+"/logP/hyperparameters/mlp_no_lc.pkl"
best_parameters_without_lc = pickle.load(open(file_path,'rb'))

# parameters
hidden_layer_sizes_list = [(150,100,50), (120,80,40) ,(100,50,30)]
max_iter_list =[5,10,50,100,200]
activation_list = ['relu', 'tanh', 'logistic']
solver_list = ['sgd','adam']
alpha_list = [0.0001, 0.05]
learning_rate_list =['constant','adaptive']

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

    best_model = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes_list[best_parameters_without_lc['hidden_layer_sizes']],
                                max_iter = max_iter_list[best_parameters_without_lc['max_iter']],
                                activation = activation_list[best_parameters_without_lc['activation']],
                                solver = solver_list[best_parameters_without_lc['solver']],
                                alpha = alpha_list[best_parameters_without_lc['alpha']],
                                learning_rate = learning_rate_list[best_parameters_without_lc['learning_rate']]
                                )

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
    all_set_df.to_excel(dirname+"/logP/results/mlp_single_no_rt.xlsx")

    print('\nSingle random run without LC information is done.')

run_best_model(des_df,lc_df)


print('\n50 repetition run without LC information is started...................................')

#### run 50 repetitions without RT
splits = 50
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

        best_model = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes_list[best_parameters_without_lc['hidden_layer_sizes']],
                                    max_iter = max_iter_list[best_parameters_without_lc['max_iter']],
                                    activation = activation_list[best_parameters_without_lc['activation']],
                                    solver = solver_list[best_parameters_without_lc['solver']],
                                    alpha = alpha_list[best_parameters_without_lc['alpha']],
                                    learning_rate = learning_rate_list[best_parameters_without_lc['learning_rate']]
                                    )

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
    final.to_excel(dirname+"/logP/results/mlp_50_no_rt.xlsx")
    
run_best_model(des_df,lc_df)
print('50 repetition run without LC information is done!!!!')


# The dataset with lc informations
#load the model without lc
path = os.getcwd()
dirname = os.path.dirname(path)
file_path = dirname+"/logP/hyperparameters/mlp_lc.pkl"
best_parameters_with_lc = pickle.load(open(file_path,'rb'))

#Single random run with RT
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

    # Single random run

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

    best_model = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes_list[best_parameters_with_lc['hidden_layer_sizes']],
                                max_iter = max_iter_list[best_parameters_with_lc['max_iter']],
                                activation = activation_list[best_parameters_with_lc['activation']],
                                solver = solver_list[best_parameters_with_lc['solver']],
                                alpha = alpha_list[best_parameters_with_lc['alpha']],
                                learning_rate = learning_rate_list[best_parameters_with_lc['learning_rate']]
                                )

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
    all_set_df.to_excel(dirname+"/logP/results/mlp_single_rt.xlsx")
    print('\nSingle random run with LC information is done.')

run_best_model(des_df,lc_df)

print('\n50 repetition run with LC information is started...........................')

# run 50 repetitions with RT
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

        best_model = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes_list[best_parameters_with_lc['hidden_layer_sizes']],
                                    max_iter = max_iter_list[best_parameters_with_lc['max_iter']],
                                    activation = activation_list[best_parameters_with_lc['activation']],
                                    solver = solver_list[best_parameters_with_lc['solver']],
                                    alpha = alpha_list[best_parameters_with_lc['alpha']],
                                    learning_rate = learning_rate_list[best_parameters_with_lc['learning_rate']]
                                    )
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
    final.to_excel(dirname+"/logP/results/mlp_50_rt.xlsx")

 
run_best_model(des_df,lc_df)
print('50 repetition run with LC information is done!!!!')

end = time.time()  # get the end time
print('\nThe total elapsed time is:', (end - start), 'S')



