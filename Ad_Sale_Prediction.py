import pandas as pd #Importing pandas library and assigning it to a variable called pd
import numpy as np #Importing numpy library and assigning it to a variable called np

#Manually importing the raw dataset file
from google.colab import files
uploaded=files.upload()

dataset=pd.read_csv('ad_dataset.csv') #Declaring the raw dataset file with the variable named dataset
print(dataset.shape) # To display the count of the dataset
print(dataset.head(5)) # To display the first 5 data from the dataset 

x=dataset.iloc[:,:-1].values # x will be assigned to all the columns except last column
y=dataset.iloc[:,-1].values # y will be assigned to only last column

#Splitting the Modal as 75% for trainning and 25% for testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Fitting the 75% of the dataset for the modal to train in the declared variable
from sklearn.preprocessing import StandarScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Modal training algorithm below
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0)
model.fit(x_train,y_train)

#Testing phase of the trained modal
y_pred=model.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Modal Accuracy Checker
from sklearn.metric import accuracy_score
print("Accuracy of the Model : {0}%".formate(accuracy_score(y_test,y_pred)*100))

#Manually Running the modal
age=int(input("Enter New Customer Age : "))
sal=int(input("Enter New Customer Salary : "))
newCust=[[age,sal]]
result=model.predict(sc.transform(newCust))
print(result)
if result==1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")
