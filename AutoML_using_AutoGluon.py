import pandas as pd
import os
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# %matplotlib notebook


##Setting Workspace and Directory

print(os.getcwd())
os.chdir(__your_location__)
print(os.getcwd())

##Reading Train data
train_data=pd.read_csv('train_data.csv')
print (train_data.shape)
train_data.head()

#Reading Test Data
test_data=pd.read_csv('test_data.csv')

#Flagging column as Test and Train
train_data['flag']='train'
test_data['flag']='test'


##Appending Train and Test for feature Normalisation
data=train_data.append(test_data)
print(train_data.shape)
print(test_data.shape)
print(data.shape)


##Since Boxcox involves non-zero values, allotting a very small value to all the columns that have 0 value
all_columns=list(data.columns)
numerical_columns=[]

for i in all_columns:
    if data[i].dtypes in ('int64','float64'):
        numerical_columns.append(i)
    
    data[i]=data[i].replace(0,0.000000001)
numerical_columns=numerical_columns[1:]

##Plotting all numerical vaeriables to see fit
for i in numerical_columns:
    # fit lognormal distribution
    g_data=sorted(list(data[i]))
    shape, loc, scale = stats.lognorm.fit(g_data, loc=0)
    pdf_lognorm = stats.lognorm.pdf(g_data, shape, loc, scale)

    # fit normal distribution
    mean, std = stats.norm.fit(g_data, loc=0)
    pdf_norm = stats.norm.pdf(g_data, mean, std)

    # fit weibull distribution
    shape, loc, scale = stats.weibull_min.fit(g_data, loc=0)
    pdf_weibull_min = stats.weibull_min.pdf(g_data, shape, loc, scale)
    
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(g_data, bins='auto', density=True)
    ax.plot(g_data, pdf_lognorm, label='lognorm')
    ax.plot(g_data, pdf_norm, label='normal')
    ax.plot(g_data, pdf_weibull_min, label='Weibull_Min')
    ax.set_xlabel(i)
    ax.set_ylabel('values')
    ax.legend();

##Scaling Creaminess and saltiness due to high skewness in the above plot
scaling_col=['Creaminess','Saltiness']
box_cox_num_col = set(numerical_columns).symmetric_difference(set(scaling_col))
box_cox_num_col = list(box_cox_num_col)

print(numerical_columns)
print(scaling_col)
print(box_cox_num_col)

##Transform Data to Boxcox
for i in box_cox_num_col:
    transformed_data, lambda_value = stats.boxcox(data[i])
    data[i+'_boxcox']=transformed_data

data=data.drop(columns=box_cox_num_col,axis=1)
print (data.shape)
data.columns

##Scaling Creaminess and saltiness due to high skewness
from sklearn.preprocessing import MinMaxScaler
for i in scaling_col:
    scaler = MinMaxScaler()
    df = data[i].values.reshape(-1, 1)
    data[i+'_scaled_feature'] = scaler.fit_transform(df)

data=data.drop(columns=scaling_col,axis=1)
print (data.shape)
data.columns       


# Label Encoding the Categorical variables
label_encoder = preprocessing.LabelEncoder()

data['Fragrance']= label_encoder.fit_transform(data['Fragrance'])
data['Fattiness']= label_encoder.fit_transform(data['Fattiness'])

train_data=data[data['flag']=='train']
test_data=data[data['flag']=='test']

print(train_data.shape)
print(test_data.shape)

##Binary Classifier

#Earmarking basic metrics
evaluation_metric="f1"
data_label="Category"
save_path="output_models_version3"

#Creating the predictor
predictor=TabularPredictor(label=data_label,path=save_path, eval_metric=evaluation_metric)

rem_cols=['ID','SubCategory','flag']
train_data=train_data.drop(rem_cols,axis=1)
predictor=predictor.fit(train_data)
predictor.leaderboard(silent=True)

#Getting Feature Importance
X = train_data
predictor.feature_importance(X)

##Tuning the Model basis the feature importance chart

print(train_data.shape)
print(test_data.shape)

#Reinitiating the Binary Classifier
evaluation_metric="f1"
data_label="Category"
save_path="output_models_version3"

train_data=data[data['flag']=='train']
test_data=data[data['flag']=='test']

predictor=TabularPredictor(label=data_label,path=save_path, eval_metric=evaluation_metric)

#Removing the columns based on feature importance cutoff 
rem_cols=['ID','SubCategory','flag','Creaminess_scaled_feature','Fattiness','Saltiness_scaled_feature','Fragrance']
train_data=train_data.drop(rem_cols,axis=1)
predictor=predictor.fit(train_data)
predictor.leaderboard(silent=True)


##Multiclass Modeling
train_data=data[data['flag']=='train']
#test_data=data[data['flag']=='test']

#Initiating Multiclass Classifier
evaluation_metric="f1_weighted"
data_label="SubCategory"
save_path="output_models_multiclass_version4"

#Initiating the classifier
predictor=TabularPredictor(label=data_label,path=save_path, eval_metric=evaluation_metric)

#train_data.columns

rem_cols=['ID','Category','flag']
train_data=train_data.drop(rem_cols,axis=1)
predictor=predictor.fit(train_data)
predictor.leaderboard(silent=True)

#Getting feature Importance
X = train_data
predictor.feature_importance(X)

# +
##Tuned Model

print(train_data.shape)
print(test_data.shape)
#Binary Classifier
evaluation_metric="f1_weighted"
data_label="SubCategory"
save_path="output_models_multiclass_version4"


train_data=data[data['flag']=='train']
test_data=data[data['flag']=='test']

predictor=TabularPredictor(label=data_label,path=save_path, eval_metric=evaluation_metric)

#Removing features basis the cutoff
rem_cols=['ID','Category','flag','Creaminess_scaled_feature','Fattiness','Saltiness_scaled_feature','Fragrance']
train_data=train_data.drop(rem_cols,axis=1)
predictor=predictor.fit(train_data)
predictor.leaderboard(silent=True)

# -

##Predicting on test set
#test_data=pd.read_csv('test_data.csv')
# how to understand word labels.
#label_encoder = preprocessing.LabelEncoder()
  # Encode labels in column 'species'.
#test_data['Fragrance']= label_encoder.fit_transform(test_data['Fragrance'])
#test_data['Fattiness']= label_encoder.fit_transform(test_data['Fattiness'])


#test_data.columns

#Predicting on the test data
multiclasspred = TabularPredictor.load("output_models_multiclass_version4/")
binarypred = TabularPredictor.load("output_models_version3/")

#Selecting the best model based the scores acheived
y_pred_proba_binary=binarypred.predict(test_data.drop(['Category','SubCategory','flag'],axis=1),model='WeightedEnsemble_L2')
y_pred_proba_multiclass=multiclasspred.predict(test_data.drop(['Category','SubCategory','flag'],axis=1),model='NeuralNetTorch')

# +
rem_cols=['Category','SubCategory']
test_data=test_data.drop(columns=rem_cols,axis=1)

test_data['Category']=y_pred_proba_binary
test_data['SubCategory']=y_pred_proba_multiclass
# -

test_data[['ID','Category','SubCategory']].to_csv('test_data_output_solo_version5.csv')


# +
print(len(y_pred_proba_binary))
      
print(len(y_pred_proba_multiclass))

print(test_data.shape)

print (test_data.columns)
# -

