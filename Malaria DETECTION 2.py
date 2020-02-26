#Malaria classification 

#Importing library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib

#Read dataset
dataset=pd.read_csv("D:/dataset.csv")
print(dataset.head())  #return top 5 data 

#Splitting to train and test set

x=dataset.drop(["Label"],axis=1)
y=dataset["Label"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Model making
classifier=RandomForestClassifier(n_estimators=100,max_depth=5)
classifier.fit(x_train,y_train)
joblib.dump(classifier,"rf_malaria_100_5")

#making predection
predicts=classifier.predict(x_test)
print(metrics.classification_report(predicts,y_test))

  