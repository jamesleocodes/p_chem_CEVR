import numpy as np
import matplotlib.pyplot as plt
# Number of pairs bars
#mse
N = 4

# Data on X-axis

# Specify the values of first bars (height)
first_bar = (0.266, 0.389, 0.255, 0.631)

# Specify the values of second bars (height)
second_bar = (0.253, 0.375, 0.247, 0.557)

# Position of bars on x-axis
ind = np.arange(N)

# Width of a bar
width = 0.3

#plotting
plt.bar(ind,first_bar ,width, label = 'Without LC informations',hatch='/////',edgecolor='black',color='white')#yerr=[0.061,0.056,0.058,0.080],capsize=6,ecolor='black')
plt.bar(ind + width, second_bar, width, label = 'With LC informations',edgecolor='black',color='black')#yerr= [0.057,0.057,0.059,0.074],capsize=6,ecolor='black')
plt.ylabel('Mean Squared Error')
plt.xticks(ind+width/2,('SVM','GB','MLP','RF'))
plt.savefig("mse.png", format='png', dpi=600)
plt.show()




#rmse

# Data on X-axis

# Specify the values of first bars (height)
first_bar = (0.513, 0.622, 0.502, 0.792 )

# Specify the values of second bars (height)
second_bar = (0.500, 0.610, 0.494, 0.744)

# Position of bars on x-axis
ind = np.arange(N)

# Width of a bar
width = 0.3

#plotting
plt.bar(ind,first_bar ,width, label = 'Without LC informations',hatch='/////',edgecolor='black',color='white')#,yerr=[0.058,0.045,0.056,0.051],capsize=6,ecolor='black')
plt.bar(ind + width, second_bar, width, label = 'With LC informations',edgecolor='black',color='black')#,yerr= [0.056,0.047,0.056,0.050],capsize=6,ecolor='black')
plt.ylabel('Root Mean Squared Error')
plt.xticks(ind+width/2,('SVM','GB','MLP','RF'))
# plt.ylim(0.40,0.90)
plt.savefig("rmse.png", format='png', dpi=600)
plt.show()


# mae

# Data on X-axis

# Specify the values of first bars (height)
first_bar = (0.352, 0.443, 0.362, 0.590)

# Specify the values of second bars (height)
second_bar = (0.348,0.435, 0.356, 0.555)

# Position of bars on x-axis
ind = np.arange(N)

# Width of a bar
width = 0.3

#plotting
plt.bar(ind,first_bar ,width, label = 'Without LC informations',hatch='/////',edgecolor='black',color='white')
            #yerr=[0.024,0.024,0.025,0.033],capsize=6,ecolor='black')
plt.bar(ind + width, second_bar, width, label = 'With LC informations',edgecolor='black',color='black')
            #yerr= [0.022,0.025,0.029,0.031],capsize=6,ecolor='black')
plt.ylabel('Mean Absolute Error')
plt.xticks(ind+width/2,('SVM','GB','MLP','RF'))
plt.savefig("mae.png", format='png', dpi=600)
plt.show()


# r2

# Data on X-axis

# Specify the values of first bars (height)
first_bar = (0.882, 0.828, 0.886, 0.722)

# Specify the values of second bars (height)
second_bar = (0.887, 0.834, 0.890, 0.754)

# Position of bars on x-axis
ind = np.arange(N)

# Width of a bar
width = 0.3

#plotting
plt.bar(ind,first_bar ,width, label = 'Without LC informations',hatch='/////',edgecolor='black',color='white')
            #yerr=[0.031,0.029,0.030,0.038],capsize=6,ecolor='black')
plt.bar(ind + width, second_bar, width, label = 'With LC informations',edgecolor='black',color='black')
            #yerr= [0.029,0.031,0.030,0.036],capsize=6,ecolor='black')
plt.ylabel('Coefficient of determination(r2)')
plt.xticks(ind+width/2,('SVM','GB','MLP','RF'))
plt.ylim(0.680,0.94)
plt.savefig("r2.png", format='png', dpi=600)
plt.show()


"""
Author: Zaw
Predicting the LogP of molecules by using SVM
"""

import numpy as np
import pandas as pd
import math
import time
import os
import sys
import shap
import warnings
warnings.simplefilter(action='ignore')
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from hyperopt import fmin,tpe,hp,STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
start = time.time()
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Rescale the feature
def scale(col):
    return (col - np.min(col))/(np.max(col)-np.min(col))

## LC informations
# Import LC informations
path = os.getcwd()
data_path = path+"/data/df_2000.csv"
col_list=['LogP','Exp_RT']
lc_df = pd.read_csv(data_path,usecols=col_list)

# Remove non_retained molecules
index=lc_df[lc_df['Exp_RT'] < 180].index
lc_df.drop(lc_df[lc_df['Exp_RT'] < 180].index,inplace=True)

# Import descriptor file
path = os.getcwd()
data_path = path+"/data/descriptors_2000.csv"
des_df = pd.read_csv(data_path,index_col=0)

# Remove non_retained molecules
des_df  = des_df.drop(des_df.index[[index]])

# data
des_without_lc = pd.concat([des_df,lc_df['LogP']],axis=1)
des_without_lc_feat_corr = des_without_lc.columns[des_without_lc.corrwith(des_without_lc['LogP']) >=0.90][:-1]
des_without_lc = des_without_lc.drop(columns=des_without_lc_feat_corr)

# Filling the nan with mean values in des_with_lc
for col in des_without_lc:
    des_without_lc[col].fillna(des_without_lc[col].mean(),inplace=True)

# Remove columns with zero vlues
des_without_lc = des_without_lc.loc[:,(des_without_lc**2).sum() != 0]

# Scale the features
no_target = des_without_lc.drop(['LogP'],axis=1)
cols = list(no_target)
no_target[cols] = no_target[cols].apply(scale,axis=0)

# data set preparation
data = pd.concat([no_target,des_without_lc['LogP']],axis=1)
X = data.drop(['LogP'],axis=1)
y = data['LogP']
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2)
model = RandomForestRegressor()
model.fit(X_train, Y_train)
shap_values = shap.TreeExplainer(model).shap_values(X_train)
f = plt.figure()
shap.summary_plot(shap_values, X_train)
# f.savefig("summary_plot_without_lc.png", bbox_inches='tight', dpi=600)

# with lc
des_with_lc = pd.concat([des_df,lc_df],axis=1)
des_with_lc_feat_corr = des_with_lc.columns[des_with_lc.corrwith(des_with_lc['LogP']) >=0.90][:-1]
des_with_lc = des_with_lc.drop(columns=des_with_lc_feat_corr)

# Filling the nan with mean values in des_with_lc
for col in des_with_lc:
    des_with_lc[col].fillna(des_with_lc[col].mean(),inplace=True)

# Remove columns with zero vlues
des_with_lc = des_with_lc.loc[:,(des_with_lc**2).sum() != 0]

# Scale the features
no_target = des_with_lc.drop(['LogP'],axis=1)
cols = list(no_target)
no_target[cols] = no_target[cols].apply(scale,axis=0)

# data set preparation
data = pd.concat([no_target,des_with_lc['LogP']],axis=1)
X = data.drop(['LogP'],axis=1)
y = data['LogP']
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2)
model = GradientBoostingRegressor()
model.fit(X_train, Y_train)
shap_values = shap.TreeExplainer(model).shap_values(X_train)
f = plt.figure()
shap.summary_plot(shap_values, X_train)
f.savefig("summary_plot_with_lc.png", bbox_inches='tight', dpi=600)


####

des_with_lc = pd.concat([des_df,lc_df],axis=1)
des_with_lc_feat_corr = des_with_lc.columns[des_with_lc.corrwith(des_with_lc['LogP']) >=0.90][:-1]
des_with_lc = des_with_lc.drop(columns=des_with_lc_feat_corr)

# Filling the nan with mean values in des_with_lc
for col in des_with_lc:
    des_with_lc[col].fillna(des_with_lc[col].mean(),inplace=True)

# Remove columns with zero vlues
des_with_lc = des_with_lc.loc[:,(des_with_lc**2).sum() != 0]

# Scale the features
no_target = des_with_lc.drop(['LogP'],axis=1)
cols = list(no_target)
no_target[cols] = no_target[cols].apply(scale,axis=0)

# data set preparation
data = pd.concat([no_target,des_with_lc['LogP']],axis=1)
X = data.drop(['LogP','LogD'],axis=1)
y = data['LogP']
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2)
model = GradientBoostingRegressor()
model.fit(X_train, Y_train)
shap_values = shap.TreeExplainer(model).shap_values(X_train)
f = plt.figure()
shap.summary_plot(shap_values, X_train)
f.savefig("summary_plot_with_lc.png", bbox_inches='tight', dpi=600)


###

des_with_lc = pd.concat([des_df,lc_df],axis=1)
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


X = data.drop(['LogP'],axis=1)
y = data['LogP']
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2)
model = GradientBoostingRegressor()
model.fit(X_train, Y_train)
shap_values = shap.TreeExplainer(model).shap_values(X_train)
f = plt.figure()
shap.summary_plot(shap_values, X_train)
f.savefig("summary_plot_with_lc.png", bbox_inches='tight', dpi=600)
