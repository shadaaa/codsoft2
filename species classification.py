#!/usr/bin/env python
# coding: utf-8

# # Use the Iris dataset to develop a model that can classify iris flowers into different species based on their sepal and petal measurements. This dataset is widely used for introductory classification tasks.

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[2]:


ds = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\archive (12)\IRIS.csv")


# In[3]:


ds


# In[5]:


ds.isnull().sum()


# In[9]:


x = ds[['sepal_length','sepal_width','petal_length','petal_width']]
y = ds[['species']]


# In[10]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


# In[14]:


model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)


# In[15]:


y_pred  = model.predict(x_test)


# In[17]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[18]:


classification_report(y_test,y_pred)


# In[19]:


def pred_species(sepall,sepalw,petall,petalw):
    scaled_inp = scaler.transform([[sepall,sepalw,petall,petalw]])
    prediction=model.predict(scaled_inp)
    return prediction[0]
try:
    sepall = float(input("Enter sepal length:"))
    sepalw = float(input("Enter sepal width:"))
    petall = float(input("Enter petal length:"))
    petalw = float(input("Enter petal width:"))
    species = pred_species(sepall,sepalw,petall,petalw)
    print(f"The predicted species is:{species}")
except ValueError:
    print("please enter valid number")


# In[ ]:





# In[ ]:




