#!/usr/bin/env python
# coding: utf-8

# ----------------------------Import all the required OOTB python libraries-------------------

# In[ ]:


from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


root_path = '/content/drive/My Drive/Maverick_Data/'
train_file = root_path+'capstone_train.csv' #Path to the training file
test_file  = root_path+'capstone_test.csv'  # path to the test file


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,recall_score, precision_score, plot_confusion_matrix
import time
from tqdm import tqdm


# In[ ]:


get_ipython().run_cell_magic(u'javascript', u'', u'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# Function: read_data_file <br />
# Input: training file path, test file path <br />
# Output : dataframes for training features, training labels, testing features, testing lables

# In[ ]:


def read_data_file(train_file,test_file):
    df_train = pd.read_csv(train_file) #reading training datafile
    df_test  = pd.read_csv(test_file)   #reading test datafile
    train_x  = df_train[df_train.columns[0:12]] #training features 
    train_y  = df_train[df_train.columns[-1]] #traininig labels
    test_x   = df_test[df_test.columns[0:12]] #training features 
    test_y   = df_test[df_test.columns[-1]] #traininig labels
    
    print('Shape of traning data :',train_x.shape,train_y.shape)
    print('Shape of testing data :',test_x.shape,test_y.shape)
    return train_x,train_y,test_x,test_y

train_x,train_y,test_x,test_y = read_data_file(train_file,test_file)


# In[ ]:


#First five rows ofthe training features
train_x.head()


# <hr>
# <h2>Performance Measurement :-</h2>

# In[ ]:


def cal_accuracy(v_test_x,v_test_y,v_model):
    #make predictions using the given model
    y_pred = v_model.predict(v_test_x)
    # how did our model perform?
    count_misclassified = (v_test_y != y_pred).sum()
    print('Total samples : {}' .format(len(v_test_x)))
    print('Misclassified samples: {}'.format(count_misclassified))
    print('Accuracy: {:.2f} %'.format(accuracy_score(v_test_y, y_pred)))
    print('\n')
    print('Classification Report :')
    print('\n')
    print(classification_report(v_test_y,y_pred,labels=np.unique(v_test_y)))
    print('Confusion Matrix :')
    print('\n')
    print(plot_confusion_matrix(v_model,v_test_x,v_test_y,cmap=plt.cm.Blues))


# <b>Exploratory Data Analysis:-</b>
#                             Exploratory Data Analysis is an approach to analyzing data sets by summarizing their main characteristics with visualizations. The EDA process is a crucial step prior to building a model in order to unravel various insights that later become important in developing a robust algorithmic model.

# In[ ]:


# Describe function with include=’all’ argument it displays the descriptive statistics for all the column
print(train_x.describe(include='all'))
#number of unique classe = 7, Therefore, we need multi-class classifier
print('\n')
print('The Target Classes :')
print(train_y.unique())
print('\n')
print('Frequncy count of each target class :')
uniqueValues, occurCount = np.unique(train_y, return_counts=True)
for i in zip(uniqueValues,occurCount):
    print(i)


# In[ ]:


#x-axis showing the specific Plant Type and the y-axis the measured value.
train_y.value_counts().plot.bar(title='Frequency distribution of Plant Type', color='orange')


# 3. The univariate distribution plot of the numerical columns which contains the histograms and the estimated PDF.

# In[ ]:


col_names = ['Heigh_From_Sea_Level','Aspect', 'Slope', 'Distance_To_Water_Source', 'Standing_Distance_To_Water_Source', 'Distance_To_Road','Shadow_In_Morning','Shadow_In_Midday','Shadow_In_Evening','Distance_To_Fire','Turf','Neighbourhood_Type']
fig, ax = plt.subplots(len(col_names), figsize=(16,12))

for i, col_val in enumerate(col_names):

    sns.distplot(train_x[col_val], hist=True, ax=ax[i])
    ax[i].set_title('Freq dist '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    ax[i].set_ylabel('Count', fontsize=8)

plt.show()


# 4. Bivariate distribution plots help us to study the relationship between two variables by analyzing the scatter plot

# In[ ]:


plant_data = train_x
plant_data= plant_data.drop([
      'Heigh_From_Sea_Level','Aspect', 'Slope', 'Distance_To_Water_Source'], axis=1)
sns.pairplot(plant_data)


# <b>Correlation matrix:-</b>
# 
# It is a table showing the value of the correlation coefficien (how strong a relationship is between two variables ) between sets of variables. Each attribute of the dataset is compared with the other attributes 
# to find out the correlation coefficient.The pairs which are highly correlated represent the same variance of the dataset thus we can further analyze them to understand which attribute among the pairs are most 
# significant for building the model.
# 
# Correlation value lies between -1 to +1. Highly correlated variables will have correlation value close to +1 and less correlated variables will have correlation value close to -1.
# 
# In this dataset, we don’t see any attributes to be correlated and the diagonal elements of the matrix value are always 1 as we are finding the correlation between the same columns thus the inference here is that all the numerical attributes are important and needs to be considered for building the model.

# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
corr  = train_x.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# <h2>Data Pre-Processing</h2>
# <b>Missing Values :-</b> </br>
# Missing values in the dataset refer to those fields which are empty or no values assigned to them.

# In[ ]:


def pre_processing_data(X):
#return true if there is any missing value in the data set
    if X.isnull().values.any() == True:
        print('There are some missing values in the training dataset ')
        print('The number of missing values in the training dataset :',X.isnull().sum())
        #The fillna function of pandas to replace 'na' values with the value 0 and inplace=True
        X.fillna(0, inplace=True)
        print('Missing values have been replaced with 0')
        print('Missing values in the dataset anymore? ',X.isnull().values.any())
        print('\n')
    else:
        print('No missing values in the data set!')
        print('\n')
    return X

#Missing values are replaced with 0, if any present in the dataset
train_x = pre_processing_data(train_x)
test_x = pre_processing_data(test_x)


# # Random Forest Classifier
# It is an ensemble tree-based learning algorithm. The Random Forest Classifier is a set of decision trees from randomly selected subset of training set. It aggregates the votes from different decision trees to decide the final class of the test object.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
t0 = time.time()
clf_rf.fit(train_x, train_y)
print('Time taken to train the Model in minutes :', (time.time() - t0)/60)
cal_accuracy(test_x,test_y,clf_rf)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_x)
# Apply transform to both the training set and the test set.
scaler_train_x = scaler.transform(train_x)
scaler_test_x = scaler.transform(test_x)
clf_scal = RandomForestClassifier()
t0 = time.time()
clf_scal.fit(scaler_train_x, train_y)
print('Time taken to train the Model in minutes :', (time.time() - t0)/60)
cal_accuracy(test_x,test_y,clf_scal)


# <h2>Logistic Regression :-</h2>
# Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable.Outputs with more than two values are modeled by multinomial logistic regression.

# In[ ]:


from sklearn.linear_model import LogisticRegression
t0 = time.time()
clf_log_reg = LogisticRegression(max_iter=1000 ).fit(train_x, train_y)
print('Time taken to train the Model in minutes :', (time.time() - t0)/60)
print('\n')
cal_accuracy(test_x,test_y,clf_log_reg)


# <h2>Normalization/ Standardization :- </h2>
# <b>Normalization</b> typically means rescales the values into a range of [0,1]. <br /> 
# <b>Standardization</b> typically means rescales data to have a mean of 0 and a standard deviation of 1 (unit variance).

# In[ ]:


#Normalize the data
from sklearn import preprocessing
normalized_train_x = preprocessing.normalize(train_x)
normalized_test_x = preprocessing.normalize(test_x)


# In[ ]:


t0 = time.time()
clf_lin_reg_norm = LogisticRegression(max_iter=1000).fit(normalized_train_x, train_y)
print('Time taken to train the Model in minutes :', (time.time() - t0)/60)
cal_accuracy(normalized_test_x,test_y,clf_lin_reg_norm)


# <b>Note :- </b>We can see that there is no significance difference in the performance after applying normalization on the data

# In[ ]:


#using scaled data
Log_reg_sca = LogisticRegression(solver = 'lbfgs')
t0 = time.time()
Log_reg_sca.fit(scaler_train_x, train_y)
print('Time taken to train the Model in minutes :', (time.time() - t0)/60)
cal_accuracy(scaler_test_x,test_y,Log_reg_sca)


# In[ ]:


from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(train_x)
train_x_robust = transformer.transform(train_x)
test_x_robust = transformer.transform(test_x)
t0 = time.time()
clf_robust= LogisticRegression(max_iter=1000).fit(train_x_robust, train_y)
print('Time taken to train the Model in minutes :', (time.time() - t0)/60)
cal_accuracy(test_x_robust,test_y,clf_robust)


# # SVM
# Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.
# 
# The advantages of support vector machines are: <br/>
# Effective in high dimensional spaces.
# Still effective in cases where number of dimensions is greater than the number of samples.
# Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
# <br/>
# Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

# In[ ]:


from sklearn.svm import LinearSVC
#one-vs-the-rest
from sklearn.feature_selection import SelectFromModel
t0 = time.time()
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_x, train_y)
print('Time taken to train the Model in minutes :', (time.time() - t0)/60)
cal_accuracy(test_x,test_y,lsvc)


# **One v/s Rest Linear Classifier**

# In[ ]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
t0 = time.time()
onevsrest_class = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_x, train_y)
print('Time taken to train the Logistic Regression Model in minutes :', (time.time() - t0)/60)
cal_accuracy(test_x,test_y,onevsrest_class)


# # Hyper Parameters Tuning

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

clf = SVC(kernel='rbf')
grid_values = {'gamma': [0.001, 0.01, 0.1, 1]}

# default metric to optimize over grid parameters: accuracy
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values, cv=3)
grid_clf_acc.fit(train_x[0:5000],train_y[0:5000])
y_decision_fn_scores_acc = grid_clf_acc.decision_function(test_x) 

print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)


# With best parameter, gamma = 0.001

# In[ ]:


from sklearn.svm import LinearSVC
#one-vs-the-rest
from sklearn.feature_selection import SelectFromModel
t0 = time.time()
lsvc = LinearSVC(C=0.001, penalty="l1", dual=False).fit(train_x, train_y)
print('Time taken to train the Model in minutes :', (time.time() - t0)/60)
cal_accuracy(test_x,test_y,lsvc)


# # K-Nearest Neighbors
# 
# In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
# <br/>
# In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.

# Finding the best number of neighbors for knn to avoid the over fitting

# In[ ]:


n_neighbors_list = [1,3,5,7,10]
for k in n_neighbors_list:    
    knn = KNeighborsClassifier(n_neighbors = k).fit(train_x, train_y)
    print(knn.score(test_x,test_y))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 
t0 = time.time()
knn = KNeighborsClassifier(n_neighbors = 3).fit(train_x, train_y)
print('Time taken to train the Model in minutes :', (time.time() - t0)/60)
cal_accuracy(test_x,test_y,knn)


# In[ ]:




