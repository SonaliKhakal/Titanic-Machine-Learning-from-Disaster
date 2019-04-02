# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:27:42 2019

@author: SONALI
"""

#importing libraries

# import the libraries
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as cr
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.externals.six import StringIO 
from IPython.display import Image
# !pip install pydotplus
import pydotplus


#Import Dataset
path = "F:/Projects/Kaggle Compitition/titanic/train.csv"
titanic = pd.read_csv(path)

titanic.head()
titanic.shape        #show number of rows and columns
titanic.info()        #dataset information
titanic.describe()    #describes numeric variables

#Check wheather missing values
titanic.isnull().sum()[titanic.isnull().sum()>0]

#check levels of object
titanic.groupby('Embarked').size()
titanic.groupby('Sex').size()
titanic.groupby('Survived').size()
titanic.groupby('Pclass').size()
titanic.groupby('SibSp').size()
titanic.groupby('Parch').size()
titanic.groupby('Ticket').size()
titanic.groupby('Cabin').size()

# Delete unnecessary columns
titanic = titanic.drop(['Name'],axis = 1)
titanic = titanic.drop(['Ticket'],axis = 1)
#delete the column Cabin (cotain greater than 75% missing values)
titanic = titanic.drop(['Cabin'],axis = 1)
titanic.shape
# Replace missing values
titanic.Embarked[titanic.Embarked.isnull()] = "S" #with unknown level

# replace with median
age_median = titanic['Age'].median()
titanic.Age[titanic.Age.isnull()] = age_median
titanic.info()


################ Data Visualization ##################
# Check data distribution
titanic.boxplot('Survived')

plt.hist(titanic.Survived)
plt.title('Survived')
plt.show()

plt.hist(titanic.Pclass)
plt.title('Pclass')
plt.show()

plt.hist(titanic.Age)
plt.title('Age')
plt.show()

plt.hist(titanic.SibSp)
plt.title('SibSp')
plt.show()

plt.hist(titanic.Parch)
plt.title('Parch')
plt.show()

plt.hist(titanic.Fare)
plt.title('Fare')
plt.show()

plt.hist(titanic.Embarked)
plt.title('Embarked')
plt.show()

plt.hist(titanic.Sex)
plt.title('Gender')
plt.show()

# Replacing levels and converting datatype to numeric
titanic.Embarked[titanic.Embarked == 'C'] = 1
titanic.Embarked[titanic.Embarked == 'Q'] = 2
titanic.Embarked[titanic.Embarked == 'S'] = 3

titanic.Sex[titanic.Sex == 'male'] = 1
titanic.Sex[titanic.Sex == 'female'] = 2

# convert the column datatype from string to numeric 
titanic[['Embarked']] = titanic[['Embarked']].apply(pd.to_numeric)
titanic[['Sex']] = titanic[['Sex']].apply(pd.to_numeric)

titanic.info()
#----------------- Model Building-------------------
####################################################

#Split train data into train1 and test1
train,test = train_test_split(titanic, test_size = 0.3)

train.shape
test.shape
train.head()

train_x = train.drop(['Survived'], axis = 1)
train_y = train.iloc[:,1:2]
test_x = test.drop(['Survived'], axis = 1)
test_y = test.iloc[:,1:2]
train_x.columns
train_y.columns
test_x.columns

train_x.shape
train_y.shape
test_x.shape
test_y.shape
type(test_y)
# random_state --> tells if the same data be used each time or different
# ---------------------------------------------------
logreg = LogisticRegression(random_state=0)
logreg.fit(train_x, train_y)

# predict on the test set
# ---------------------------------------------------
pred_y = logreg.predict(test_x)

# cf = confusion_matrix(test_y, pred_y, labels=['actual','predicted'])
labels=[0,1]
cf = confusion_matrix(pred_y,test_y,labels)
print(cf)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(test_x, test_y)))

# confusion matrix with details
# -----------------------------
ty=test_y['Survived'].tolist()
py=list(pred_y)
cm1=ConfusionMatrix(py,ty)
print(cm1)
cm1.print_stats()
cm1.plot()

# Classification report : precision, recall, F-score
# ---------------------------------------------------
print(cr(test_y, pred_y))

################ Decision Tree 
#######################################################
#------------------------------------------------------
# Model 1) DT with gini index criteria ---> hyperparameter
# ---------------------------------------------------------
clf_gini = dtc(criterion = "gini", random_state = 100, 
               max_depth=3, min_samples_leaf=5)

fit1 = clf_gini.fit(train_x, train_y)
print(fit1)

# dotfile="F:/aegis/6_python/practice/ml/2 classification/trees/dt/dt_gini.dot"
# tree.export_graphviz(fit1,out_file=dotfile)
# cut and paste the code from this file into http://webgraphviz.com
# to view the graph

# tree visualisation
# -------------------------------------
dot_data = StringIO()

tree.export_graphviz(fit1, out_file=dot_data,  
                filled=True, rounded=True, special_characters=True)

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())

# predictions
# -------------------------------------
pred_gini = fit1.predict(test_x)
pred_gini
len(test_y)
len(pred_gini)
print("Gini Accuracy is ", 
      accuracy_score(test_y,pred_gini)*100)

# create dataframe with the actual and predicted results
# -------------------------------------------------------
df_results1 = pd.DataFrame({'actual':test_y, 
                            'predicted':pred_gini})
df_results1

# confusion matrix of the Y-variables for both models
# --------------------------------------------------
confusion_matrix(test_y, pred_gini)

# another nice way to plot the results
# -------------------------------------
dtcm1=ConfusionMatrix(test_y, pred_gini)
dtcm1
# plot
# -------------------------------------
dtcm1.plot()
dtcm1.print_stats()

# plotting the results ??
# ---------------------

# select the best columns (Feature selection) using RFE
# -----------------------------------------------------
#from sklearn.feature_selection import RFE
rfe = RFE(fit1, 5)  #top 5 features
rfe = rfe.fit(test_x, test_y)
support = rfe.support_
ranking = rfe.ranking_


# Model 2) DT with Entropy(Information Gain) criteria
# ----------------------------------------------------
clf_entropy=dtc(criterion="entropy", 
                random_state=100, max_depth=3, 
                min_samples_leaf=5)

fit2 = clf_entropy.fit(train_x,train_y)
print(fit2)

pred_entropy = fit2.predict(test_x)

pred_entropy

len(test_y)
len(pred_entropy)

print("Entropy Accuracy is ", 
      accuracy_score(test_y,pred_entropy)*100)

df_results2 = pd.DataFrame({'actual':test_y, 
                            'predicted':pred_entropy})
df_results2

# confusion matrix of the Y-variables
# ------------------------------------
dtcm2 = confusion_matrix(test_y, pred_entropy)

dtcm2

##################################################
# Logistic Regression is better 
# Predicting Test dataset with Logistic Regression model

path = "F:/Projects/Kaggle Compitition/titanic/test.csv"
testd = pd.read_csv(path)

testd.shape

# Replacing levels and converting datatype to numeric
testd.Embarked[testd.Embarked == 'C'] = 1
testd.Embarked[testd.Embarked == 'Q'] = 2
testd.Embarked[testd.Embarked == 'S'] = 3

testd.Sex[testd.Sex == 'male'] = 1
testd.Sex[testd.Sex == 'female'] = 2

# convert the column datatype from string to numeric 
testd[['Embarked']] = testd[['Embarked']].apply(pd.to_numeric)
testd[['Sex']] = testd[['Sex']].apply(pd.to_numeric)

testd.info()

# Delete unnecessary columns
testd = testd.drop(['Name'],axis = 1)
testd = testd.drop(['Ticket'],axis = 1)
#delete the column Cabin (cotain greater than 75% missing values)
testd = testd.drop(['Cabin'],axis = 1)

# replace with median
age_median = testd['Age'].median()
testd.Age[testd.Age.isnull()] = age_median
fare_median = testd['Fare'].median()
testd.Fare[testd.Fare.isnull()] = fare_median
testd.info()
testd.columns

pred_y = logreg.predict(testd)
pID = testd['PassengerId']
predic_test = pd.DataFrame({'PassengerId':pID, 'Survived':pred_y})
print(predic_test)

# convert in .csv file
predic_test.to_csv("F:/Projects/Kaggle Compitition/titanic/prediction.csv")
