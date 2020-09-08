# =============================================================================
# Importing libraries&datasets
# =============================================================================

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

#Importing the dataset
test = pd.read_csv("test.csv")
dataset = pd.read_csv("train.csv")
pd.set_option('display.max_columns', 20)
np.set_printoptions(threshold = sys.maxsize)
print('Train shape: {}, \nTest shape: {}\n'.format( dataset.shape, test.shape))

#data types
print('Data types:\n', dataset.dtypes, '\n')

dataset.head()
dataset.tail()

# =============================================================================
# Data Exploration
# =============================================================================

#Descriptive statistics 
print('Descriptive statistics:\n', dataset.describe(), '\n')

#skewness&kurtosis
from scipy.stats import skew, kurtosis
numerical_columns = [x for x in dataset.dtypes.index
                     if dataset.dtypes[x] != 'object']
numerical_columns = [x for x in numerical_columns
                     if x not in ['Loan_Amount_Term', 'Credit_History']]

for column in numerical_columns:
    print('Variable {}'.format(column))
    print('Skewness: {}'.format(dataset[column].skew().round(2)))
    print('Kurtosis: {}\n'.format(dataset[column].kurtosis().round(2)))

#Categorical unique variables
print('Unique values:\n', dataset.apply(lambda x: len(x.unique())))

#Frequency of categories for categorical variables
categorical_columns = [x for x in dataset.dtypes.index 
                       if dataset.dtypes[x]=='object']
categorical_columns = [x for x in categorical_columns
                       if x not in ['Loan_ID', 'source']]
categorical_columns = categorical_columns + ['Loan_Amount_Term', 'Credit_History']

for column in categorical_columns:
    print("\nFrequency of Categories for variable {}".format(column))
    print (dataset[column].value_counts())

# =============================================================================
# Univariate analysis - plots
# =============================================================================

for variable in numerical_columns:
    plt.subplot(121)
    sns.distplot(dataset[variable]);
    plt.subplot(122)
    dataset[variable].plot.box(figsize = (16, 5))
    plt.show()

#Plotting categorical and ordinal variables 
for variable in categorical_columns:
    dataset[variable].value_counts(normalize = True, ascending = True).plot.bar()
    plt.xlabel(variable)
    plt.ylabel('%')
    plt.title('Percentage distribution')
    plt.show()

# =============================================================================
# #Bi-variate analysis
# =============================================================================

for variable in categorical_columns:
    cross = pd.crosstab(index = dataset[variable], 
                        columns = dataset['Loan_Status'])
    cross.div(cross.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
    print('\n', cross, '\n')

for variable in numerical_columns:
    dataset.boxplot(column = variable, by = 'Loan_Status')
    plt.suptitle('')

#Comparing education with dependents
cross = pd.crosstab(index = dataset['Dependents'], 
                        columns = dataset['Education'])
cross.div(cross.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)

#Comparing education with property area
cross = pd.crosstab(index = dataset['Property_Area'], 
                        columns = dataset['Education'])
cross.div(cross.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)

#Income vs education
dataset.boxplot(column = 'ApplicantIncome', by = 'Education')
plt.suptitle('')    

#Income vs property area
dataset.boxplot(column = 'ApplicantIncome', by = 'Property_Area')
plt.suptitle('')    

#Loan Amount vs education - interesting
dataset.boxplot(column = 'LoanAmount', by = 'Education')
plt.suptitle('')    

#Creating a total income variable
dataset['Total_Income'] = dataset['ApplicantIncome']+dataset['CoapplicantIncome']
dataset.boxplot(column = 'Total_Income', by = 'Loan_Status')
plt.suptitle('')
plt.figure()
sns.distplot(dataset['Total_Income'])

#Correlation matrix
corrMatrix = dataset.corr()
plt.figure()
sns.heatmap(corrMatrix, annot=True)

# =============================================================================
# #Missing value treatment
# =============================================================================
#Train set
print('\nMissing values:\n', dataset.apply(lambda x: sum(x.isnull())))

#filling na's with mode for categorical variables
for variable in categorical_columns:
    dataset[variable].fillna(dataset[variable].mode() [0], inplace = True)

dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace = True)

#Checking the results
print('\nMissing values:\n', dataset.apply(lambda x: sum(x.isnull())))

#Taking care of the test set
print('\nMissing values:\n', test.apply(lambda x: sum(x.isnull())))

cat_col2 = [x for x in dataset.dtypes.index
            if dataset.dtypes[x]=='object']
cat_col2 = [x for x in categorical_columns
                       if x not in ['Loan_ID', 'Loan_Status', 'source']]

for variable in cat_col2:
    test[variable].fillna(dataset[variable].mode() [0], inplace = True)

test['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace = True)

#Checking the results
print('\nMissing values:\n', test.apply(lambda x: sum(x.isnull())), 'n')

# =============================================================================
# #Outlier treatment
# =============================================================================

#Loan Amount
dataset['LoanAmount_log'] = np.log(dataset['LoanAmount'])
plt.figure()
dataset['LoanAmount_log'].hist(bins = 20)
plt.title('LoanAmount_log (training set)')
plt.show()

#Loan Amount log
test['LoanAmount_log'] = np.log(test['LoanAmount'])
test['LoanAmount_log'].hist(bins = 20)
plt.title('LoanAmount_log (test set)')
plt.show()

#Total Income train
dataset['TotalIncome_log'] = np.log(dataset['Total_Income'])
dataset['TotalIncome_log'].hist(bins = 20)
plt.title('TotalIncome_log (training set)')
plt.show()

#Total Income test
test['Total_Income'] = test['ApplicantIncome']+test['CoapplicantIncome']
test['TotalIncome_log'] = np.log(test['Total_Income'])
test['TotalIncome_log'].hist(bins = 20)
plt.title('TotalIncome_log (test set)')
plt.show()

# =============================================================================
# Data preprocessing
# =============================================================================

#Training set
dataset['Loan_Status'] = dataset['Loan_Status'].map(dict(Y=1, N=0))
X = dataset.drop(['Loan_ID', 'Loan_Status'], 1).values
y = dataset['Loan_Status'].values

#Test set
X_final_test = test.drop('Loan_ID', 1).values

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1, 2, 3, 4, 10])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

#Dividing the training set further into training and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, [12, 13, 14, 15, 17, 18, 19]] = sc.fit_transform(X_train[:, [12, 13, 14, 15, 17, 18, 19]])
X_test[:, [12, 13, 14, 15, 17, 18, 19]] = sc.transform(X_test[:, [12, 13, 14, 15, 17, 18, 19]])

# =============================================================================
# Building the models
# =============================================================================

#importing necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

#Creating a function for estimating models and evaluating the outcomes
def classifier(method):
    
    #fitting the model
    model = method.fit(X_train, y_train)
    print(method, '\n')
    #Predicting the test set results
    y_pred = model.predict(X_test)
    print('Correct predictions: {}\nIncorrect predictions: {}\n'.format(sum(y_pred == y_test), 
          sum(y_pred != y_test)))
    
    #Making a confusion matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix: \n{}'.format(cm),
          '\n')
    print('Accuracy score: {:.4f}'.format(accuracy_score(y_test, y_pred)),
          '\n')
    
    #Calculating sensitivity and specificity
    spec = cm[0,0]/(cm[0,0]+cm[0,1])
    sens = cm[1,1]/(cm[1,0]+cm[1,1])
    prec = cm[1,1]/(cm[0,1]+cm[1,1])
    print('Specificity: {:.2f} \nSensitivity: {:.2f} \nPrecision: {:.2f}'.format(spec, sens, prec))
    
    #Calculating AUC
    from sklearn.metrics import roc_auc_score
    print('AUC score: {:.4f}\n'.format(roc_auc_score(y_test, y_pred)))
    
    #Performing k-fold cross validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(model, X_train, y_train, cv = 10)
    print('Accuracy (10-fold cross validation): {:.2f}'.format(accuracies.mean()*100))
    print('Standard deviation accuracy (10-fold cross validation): {:.2f}\n'.format(accuracies.std()*100))
   

#Building models and evaluating the results
classifier(LogisticRegression(max_iter = 200))
classifier(KNeighborsClassifier())
classifier(SVC()) #rbf kernel
classifier(SVC(kernel = 'sigmoid'))
classifier(SVC(kernel = 'poly'))
classifier(GaussianNB())
classifier(DecisionTreeClassifier(criterion = 'entropy'))
classifier(RandomForestClassifier(criterion = 'entropy')) 
classifier(XGBClassifier(scale_pos_weight = 100))
classifier(CatBoostClassifier(logging_level = 'Silent'))

# =============================================================================
# Grid search
# =============================================================================

#Logistic regression
from sklearn.model_selection import GridSearchCV
params_lr = [{'penalty': ['elasticnet', 'none'], 'C': [0.25, 0.5, 0.75, 1], 'solver': ['saga'], 'max_iter': [125, 150, 175, 200]},
             {'penalty': ['l2', 'none'], 'C': [0.25, 0.5, 0.75, 1], 'solver': ['newton-cg', 'sag', 'lbfgs'], 'max_iter': [125, 150, 175, 200]},
             {'penalty': ['l1', 'none'], 'C': [0.25, 0.5, 0.75, 1], 'solver': ['liblinear', 'saga'], 'max_iter': [125, 150, 175, 200]}]
grid_search = GridSearchCV(estimator = LogisticRegression(), 
                           param_grid = params_lr,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# =============================================================================
# Summary
# =============================================================================

#Final model
model = LogisticRegression(C = 0.75, max_iter = 125, penalty = 'l1', solver = 'liblinear')
classifier(model)



