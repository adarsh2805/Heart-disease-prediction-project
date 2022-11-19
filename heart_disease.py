import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import sklearn


pd.set_option('display.max_columns',10000)
pd.set_option('display.max_row',80)

datapath=os.path.join(os.path.abspath(os.path.dirname(os.path.curdir)),'dataset\Heart_Disease_Prediction.csv')
dataframe = pd.read_csv(datapath)

# #checking null values
# print(dataframe.isnull().sum())
# print(dataframe.describe())
# print(dataframe.info())

# #visualizing data

#hist map
# dataframe.hist()
# plt.tight_layout()
# plt.show()
y_feature=dataframe.iloc[:,-1]
x_feature=dataframe.iloc[:,0:-1]

# #checking how many having diseased and not diseased people
# plt.title('COUNTED NUMBER OF PEOPLE HAVE DISEASE')
# sns.countplot(x='Heart Disease',data=dataframe,order=['Absence','Presence'],dodge=True)
# plt.grid()
# plt.show()

# print(dataframe)

#pairplot
# plt.title('HEART DISEASE COULMN COMP WITH ALL OTHER FIELD')
sns.pairplot(dataframe,hue='Heart Disease')
plt.tight_layout()
plt.show()

row=4
col=5
i=1
for column in x_feature.columns:
    plt.subplot(row,col,i)
    plt.title('status vs ' + str(column))
    bval=sns.barplot(x='Sex',y=str(column),data=dataframe,ci=20,hue='Heart Disease')
    plt.xlabel('GENDER')
    
    # for remove legend
    bval.legend_.remove()
    plt.tick_params(labelsize=5)
    i+=1


plt.tight_layout()

plt.show()

#one hot coding

# sns.heatmap(dataframe.corr(),annot=True)
# plt.show()

y_feature=pd.get_dummies(dataframe,drop_first=True)
y_feature= y_feature['Heart Disease_Presence']


print(x_feature.shape,y_feature.shape)

# #split the data for train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_feature,y_feature,random_state=42529,stratify=y_feature,test_size=0.4)

print(y_train.shape,y_test.shape)
#convert the data into standard scaller 
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
x_data=StandardScaler()
x_test=x_data.fit_transform(x_test)
x_train=x_data.fit_transform(x_train)



#select the model

#desion tree
from sklearn.tree import DecisionTreeClassifier

des = DecisionTreeClassifier(random_state=123)
des.fit(x_train,y_train)
y_pred=des.predict(x_test)


#SVM
from sklearn.svm import SVC

svm=SVC(kernel='rbf',random_state=123)
svm.fit(x_train,y_train)
svcrespre=svm.predict(x_test)

#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
rdf=RandomForestClassifier(random_state=1234)
rdf.fit(x_train,y_train)
print(y_train.shape)
print('#'*60)
rdres=rdf.predict(x_test)

#logistic regression 
from sklearn.linear_model import LogisticRegression
lra=LogisticRegression()
lra.fit(x_train,y_train)
pred_lra=lra.predict(x_test)

#find the accuracy 

print('&'*30+'ACCURACY FINDING :'+'&'*30)

from sklearn.metrics import accuracy_score
print("DESITION TREE :",accuracy_score(y_test,y_pred))
print("SVM :",accuracy_score(y_test,svcrespre))
print("RANDOMFOREST :",accuracy_score(y_test,rdres))
print("Logistic regression :",accuracy_score(y_test,pred_lra))




#cross validation
from sklearn.model_selection import cross_validate
print("\n"*2)
print('$'*30+'CROSS VALIDATION '+'&'*30)
mdl_rpt=[]
list_model=[des,svm,rdf,lra]
for det in list_model:
    res=cross_validate(det,x_train,y_train,scoring='accuracy',cv=30,return_train_score=True)
    mdl_rpt.append(pd.Series(res['test_score']).mean())
    print(f"{det} " ,pd.Series(res['test_score']).mean())

df_mdl_result=pd.DataFrame({'ALGORITHM NAME':['DECISION TREE','RANDOM FOREST','SVM','LOGISTIC REGRESSION'],'ACCURACY':mdl_rpt})
print(df_mdl_result)

# sns.barplot(x='ALGORITHM NAME',y='ACCURACY',data=df_mdl_result)
# plt.show()

#hyperparameter tunning

# from sklearn.model_selection import GridSearchCV

#desion tree :

# drfc = {'criterion':["gini", "entropy"],
#          'splitter':['best'],
#         'max_depth':[5,6,7]}

# res = GridSearchCV(estimator=des,param_grid=drfc,scoring='accuracy',cv=30,return_train_score=True)
# fin_res=res.fit(x_train,y_train)
# print(fin_res.cv_results_)

# #svm

# srfc ={'kernal':['rbf','linear','poly'],
#         'degree':5,
#           'gama':0.5}


#evaluating randomforest

from sklearn.metrics import confusion_matrix,classification_report
conf=confusion_matrix(y_true=y_test,y_pred=svcrespre)
# plt.title('CONFUTION MATRIX')

conf=classification_report(y_test,svcrespre)
# sns.heatmap(conf,annot=True)
# plt.show()











