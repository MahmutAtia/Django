import pandas as pd
from sklearn import linear_model
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


data = pd.read_csv("data.csv")
data.drop("Unnamed: 0" ,inplace=True, axis=1)
y,x = data.iloc[:,2],data.iloc[:,:2]


x_train,x_text, y_train,y_test = train_test_split(x,y,test_size= 0.3,shuffle=True)

lm = linear_model.LinearRegression()
lm.fit(x_train,y_train)


print(lm.score(x_text,y_test))

pickle.dump(lm, open("model.pkl", "wb"))