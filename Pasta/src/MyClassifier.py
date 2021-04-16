# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import argparse
import os


# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# %% [markdown]
# # Loading file and arguments checking helper functions

# %%
class CheckAlgorithmAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values not in ['NN','LR','NB','DT','BAG','ADA','GB','RF','SVM','P']:
            raise ValueError("Please select a valid action : NN,LR,NB,DT,BAG,ADA,GB,RF,SVM,P")
        setattr(namespace, self.dest, values)


# %%
class DatasetAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        from os.path import exists
        if not exists(values):
            raise ValueError("Cannot find dataset! Please provide valid dataset path in first argument")
        setattr(namespace, self.dest, values)


# %%
def loadCsv(datasetFilename):
    
    def convertClass(value):
        if value == 'class1':
            return 0
        else: 
            if value == 'class2':
                 return 1
            else:
                raise ValueError('Class not defined for some rows')

    converters = { 'class': convertClass }

    df = pd.read_csv(datasetFilename, header=0, sep=',')
    
    return df


# %%


# %% [markdown]
# # Dataset preprocessing
# 
# You will need to pre-process the dataset, before you can apply the classification algorithms. Three types of pre-processing are required:    
#     
# 1. The missing attribute values should be replaces with the mean value of the column using 
# sklearn.impute.SimpleImputer.   
# 
# 2. Normalisation of each attribute should be performed using a min-max scaler to normalise the 
# values between [0,1] with sklearn.preprocessing.MinMaxScaler.   
# 
# 3. The classes class1 and class2 should be changed to 0 and 1 respectively.   
# 
# 4. The value of each attribute should be formatted to 4 decimal places using .4f.   
# 

# %%
######################
# DATA PREPROCESSING
######################

def datasetPreProcessing(df):

    df.replace('?', np.NaN, inplace=True)
    #df['class'].replace(['class1','class2'],[0,1],inplace=True) doesn't work in PASTA
    df['class']=df['class'].map({'class1':0, 'class2':1})
    #df.dropna(subset=['class'], inplace=True) tried for pasta

    y = df.iloc[:,-1:]
    # y=pd.DataFrame(df.iloc[:,-1:])
    # y.replace(['class1','class2'],[0,1],inplace=True)
    # y2=y.iloc[:, -1:]
    
    imputer = SimpleImputer(missing_values=np.NaN, strategy='mean', copy=False)
    
    X = df.iloc[:,:-1]
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    scaler = MinMaxScaler(copy=False)
    X_minmax = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)
    
    return X_minmax, y

############################################
## Print processed dataset
#############################################
def processP(X, y):
    
    df = pd.concat([X, y], axis=1)
        
    for row_index, features in df.iterrows():
        for i, feature in enumerate(features):
            if i < len(features)-1:
                print('{:.4f},'.format(feature), end='')
            else:
                print('{:d}'.format(int(feature)))




# %% [markdown]
# ## *2 Implementing multiple classifiers to the pre-processed dataset*
# ### Nearest Neighbor, Logistic Regression, Naïve Bayes, Decision Tree, Bagging, Ada Boost and Gradient Boosting
# 
# Evaluate performance of each classifier  using 10-fold cross validation using sklearn.model_selection.StratifiedKFold with these options:
# ``` cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0) ```
# 
# Notes:    
#  - print the mean accuracy score with precision of 4 decimal places using .4f
#  -  no new line character after the accuracy ```print(data, end=’’)```
# 

# %%
######################################################
#using  KNeighboursClassifier from sklearn.neighbours
######################################################
def kNNClassifier(X, y, K):
    from sklearn.neighbors import KNeighborsClassifier
    
    classifier=KNeighborsClassifier(n_neighbors=K)
    
    cvKFold10=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    
    scores = cross_val_score(classifier, X, y.squeeze(), cv=cvKFold10)
    
    return scores, scores.mean()



# %%
######################################################
#using GaussianNB from sklearn.naive_bayes
######################################################
def nbClassifier(X, y):
    from sklearn.naive_bayes import GaussianNB
    
    classifier = GaussianNB()
    cvKFold10=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y.squeeze(), cv=cvKFold10)
    
    return scores, scores.mean()




# %%
######################################################
#using  LogisticRegression from sklearn.linear_model.
######################################################
def logregClassifier(X, y):
    from sklearn.linear_model import LogisticRegression
    
    classifier = LogisticRegression(random_state=0)
    cvKFold10=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y.squeeze(), cv=cvKFold10)
    
    return scores, scores.mean()




# %%
##############################################################################################
#using DecisionTreeClassifier from sklearn.tree, with information gain (the entropy criterion)
###############################################################################################
def dtClassifier(X, y):
    from sklearn.tree import DecisionTreeClassifier
    
    classifier=DecisionTreeClassifier(criterion='entropy', random_state=0)
    cvKFold10=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y.squeeze(), cv=cvKFold10)
    
    return scores, scores.mean()




# %%
####################################################################################################
#using BaggingClassifier from sklearn.ensemble. should combine Decision Trees with information gain.
####################################################################################################
def bagDTClassifier(X, y, n_estimators, max_samples, max_depth):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=0), 
                                   n_estimators=n_estimators, max_samples=max_samples, random_state=0)
    cvKFold10=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y.squeeze(), cv=cvKFold10)
    
    return scores, scores.mean()




# %%
######################################################################################################
#using AdaBoostClassifier from sklearn.ensemble. should combine Decision Trees with information gain.
######################################################################################################
def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth):
    from sklearn.ensemble import AdaBoostClassifier

    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, random_state=0), n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    
    cvKFold10=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y.squeeze(), cv=cvKFold10)
    
    return scores, scores.mean()




# %%
##############################################################################################################
#using GradientBoostingClassifier from sklearn.ensemble. should combine Decision Trees with information gain.
##############################################################################################################
def gbClassifier(X, y, n_estimators, learning_rate):
    from sklearn.ensemble import GradientBoostingClassifier

    classifier = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)

    cvKFold10=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y.squeeze(), cv=cvKFold10)
    
    return scores, scores.mean()



# %% [markdown]
# ## *3 Parameter Tuning*
# 
# For two other classifiers, Linear SVM and Random Forest, we would like to find the best parameters using grid search with 10-fold stratified cross validation (```GridSearchCV``` in sklearn).    
# The split into training and test subsets should be done using train_test_split from ```sklearn.model_selection``` with stratification and ```random_state=0``` (as in the tutorials but with
# random_state=0)   
# %% [markdown]
# ### 3.1 Best linear classifier
# 
# For Linear SVM, your program should output exactly 4 lines:      
# The first line contains the optimal C value,     
# the second line contains the optimal gamma value,   
# and the third line contains the best cross-validation accuracy score formatted to 4 decimal places using .4f   
# and the fourth line contains the test set accuracy score also formatted to 4 decimal places.    
# For instance, if the optimal C and gamma values are 0.001 and 0.1 respectively, and the two accuracies are 0.999874 and 0.952512, your program output should look like:
# 
# 0.001   
# 0.1    
# 0.9999    
# 0.9525   
# 

# %%
#############################
# use SVC from sklearn.svm.
#############################
def bestLinClassifier(X,y):
    
    #setting up the parameter selection grid
    C = [0.001, 0.01, 0.1, 1, 10, 100]
    gamma = [0.001, 0.01, 0.1, 1, 10, 100]
    
    param_grid = {'C': C,'gamma': gamma}
    
    # split between training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y.squeeze(), stratify=y, random_state=0)


    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    
    classifier = SVC(random_state=0)
    
    cvKFold10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    
    grid_search = GridSearchCV(classifier, param_grid, cv=cvKFold10)

    #use the training data to look for the best parameters from the grid using 10 fold cross-validation 
    grid_search.fit(X_train, y_train)

    # print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
    # print("Best parameters: {}".format(grid_search.best_params_))
    # print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    # print("Best estimator:\n{}".format(grid_search.best_estimator_))
    
    # print best parameters found
    # print best cross-validation accuracy score
    # print best test set accuracy score
    # (see Section 4)
    
    return { 'best_parameters': {'C': grid_search.best_params_['C'], 'gamma': grid_search.best_params_['gamma']},
             'best_cross_val_accuracy_score': grid_search.best_score_,
             'best_test_set_accuracy_score': grid_search.score(X_test, y_test)
           }




# %% [markdown]
# ### 3.1 Best random forest classifier
# 
# For Random Forest, your program should output exactly 4 lines.   
# The first line contains the optimal n_estimators,    
# the second line contains the optimal max_leaf_nodes,    
# the third line contains the best cross validation accuracy score truncated to 4 decimal places using .4f    
# and the fourth line contains the test set accuracy score also truncated to 4 decimal places.    
# 

# %%
##########################################################################################################
#use RandomForestClassifier from sklearn.ensemble with information gain and max_features set to ‘sqrt’.
##########################################################################################################
def bestRFClassifier(X,y):
    
    #setting up the parameter selection grid
    n_estimators = [10, 30]
    max_leaf_nodes = [4, 16]
    
    param_grid = {'n_estimators': n_estimators,'max_leaf_nodes': max_leaf_nodes, 'max_features': ['sqrt']}
    
    # split between training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y.squeeze(), stratify=y, random_state=0)


    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    classifier = RandomForestClassifier(criterion='entropy', random_state=0)
    
    cvKFold10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    
    grid_search = GridSearchCV(classifier, param_grid, cv=cvKFold10)

    #use the training data to look for the best parameters from the grid using 10 fold cross-validation 
    grid_search.fit(X_train, y_train)

    # print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
    # print("Best parameters: {}".format(grid_search.best_params_))
    # print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    # print("Best estimator:\n{}".format(grid_search.best_estimator_))
    
    # print best parameters found
    # print best cross-validation accuracy score
    # print best test set accuracy score
    # (see Section 4)
    
    return { 'best_parameters': {'n_estimators': grid_search.best_params_['n_estimators'], 'max_leaf_nodes': grid_search.best_params_['max_leaf_nodes']},
             'best_cross_val_accuracy_score': grid_search.best_score_,
             'best_test_set_accuracy_score': grid_search.score(X_test, y_test)
           }




# %%
# PERFORMANCE MEASURES
# https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

# %% [markdown]
# # MAIN

# %%
# https://stackoverflow.com/questions/4480075/argparse-optional-positional-arguments
# https://dusty.phillips.codes/2018/08/13/python-loading-pathlib-paths-with-argparse/

def main():
    
    
    # ARGUMENTS PARSE AND VALIDATION
    
    parser = argparse.ArgumentParser(description='Run some ML algorithm on the Breast cancer detection Wisconsin dataset')
    parser.add_argument('datasetFilename',  metavar='datasetFilename', type=str, default='breast-cancer-wisconsin.csv', help='path of the dataset', action=DatasetAction)
    parser.add_argument('algorithmName', metavar='algorithmName', type=str, default='NN',help='algorithm selection', action=CheckAlgorithmAction)
    parser.add_argument('parameterFilename', nargs='?',  metavar='parameterFilename', type=str, help='path of parameters', default='')
    p = parser.parse_args()
    
    if p.algorithmName in ['NN', 'BAG', 'ADA','GB']:
        if not os.path.exists(p.parameterFilename):
            raise ValueError('Cannot find parameter file. Please provide a valid parameter file path')
       
    if p.algorithmName in ['NN', 'BAG', 'ADA','GB']:
        parametersDf = pd.read_csv(p.parameterFilename, header=0)
        parametersDf_headers = parametersDf.columns.tolist()

        if p.algorithmName == 'NN' and parametersDf_headers != ['K']:
            raise ValueError("Invalid CSV file header {}, should be K".format(*parametersDf_headers))
            
        if p.algorithmName == 'BAG' and parametersDf_headers != ['n_estimators','max_samples','max_depth']:
            raise ValueError("Invalid CSV file header {}, should be n_estimators,max_samples,max_depth".format(*parametersDf_headers))
            
        if p.algorithmName == 'ADA' and parametersDf_headers != ['n_estimators','learning_rate','max_depth']:
            raise ValueError("Invalid CSV file header {}, should be n_estimators,learning_rate,max_depth".format(*parametersDf_headers))
            
        if p.algorithmName == 'GB' and parametersDf_headers != ['n_estimators','learning_rate']:
            raise ValueError("Invalid CSV file header {}, should be n_estimators,learning_rate".format(*parametersDf_headers))
            
    # DATA PREPROCESSING
    #--------------------
    
    
    df = loadCsv(p.datasetFilename)
    
    X, y = datasetPreProcessing(df)
    
    # CROSS VALIDATION SCORES FOR VARIOUS ALGORITHMS
    #------------------------------------------------
                   
    if p.algorithmName == 'P': #P for printing the pre-processed dataset
        processP(X, y)
        return
        
    if p.algorithmName == 'NN': #NN for Nearest Neighbour.
        scores, scores_mean = kNNClassifier(X, y, parametersDf['K'][0])
        print('{:.4f}'.format(scores_mean), end='')
        return

    if p.algorithmName == 'LR':  #LR for Logistic Regression
        scores, scores_mean = logregClassifier(X, y)
        print('{:.4f}'.format(scores_mean), end='')
        return
        
    if p.algorithmName == 'NB': #NB for Naïve Bayes
        scores, scores_mean = nbClassifier(X, y)
        print('{:.4f}'.format(scores_mean), end='')
        return

    if p.algorithmName == 'DT': #DT for Decision Tree
        scores, scores_mean = dtClassifier(X, y)
        print('{:.4f}'.format(scores_mean), end='')
        return
        
    if p.algorithmName == 'BAG': #BAG for Ensemble Bagging DT
        scores, scores_mean = bagDTClassifier(X, y, parametersDf['n_estimators'][0], parametersDf['max_samples'][0],parametersDf['max_depth'][0])
        print('{:.4f}'.format(scores_mean), end='')
        return

    if p.algorithmName == 'ADA': #ADA for Ensemble ADA boosting DT
        scores, scores_mean = bagDTClassifier(X, y, parametersDf['n_estimators'][0], parametersDf['learning_rate'][0],parametersDf['max_depth'][0])
        print('{:.4f}'.format(scores_mean), end='')
        return
        
    if p.algorithmName == 'GB': #GB for Ensemble Gradient Boosting
        scores, scores_mean = gbClassifier(X, y, parametersDf['n_estimators'][0], parametersDf['learning_rate'][0])
        print('{:.4f}'.format(scores_mean), end='')
        return
    
    # PARAMETER SELECTIONS FOR RF AND SVM
    #------------------------------------------------

    if p.algorithmName == 'RF': #RF for Random Forest
        bestRF = bestRFClassifier(X, y)
        print('{:d}'.format(bestRF['best_parameters']['n_estimators']))
        print('{:d}'.format(bestRF['best_parameters']['max_leaf_nodes']))
        print('{:.4f}'.format(bestRF['best_cross_val_accuracy_score']))
        print('{:.4f}'.format(bestRF['best_test_set_accuracy_score']), end='')
        return
        
    if p.algorithmName == 'SVM': #SVM for Linear SVM
        bestSVC = bestLinClassifier(X, y)
        print('{}'.format(bestSVC['best_parameters']['C']))
        print('{}'.format(bestSVC['best_parameters']['gamma']))
        print('{:.4f}'.format(bestSVC['best_cross_val_accuracy_score']))
        print('{:.4f}'.format(bestSVC['best_test_set_accuracy_score']), end='')
        return



# %%
if __name__ == "__main__":
    main()


# %%



