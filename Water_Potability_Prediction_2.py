#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers, Sequential, Model
import joblib
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df= pd.read_csv('water_potability_2.csv')


# In[3]:


df.head(10)


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df=df.dropna()
df.head(10)


# In[8]:


#df['ph']=df['ph'].fillna(df['ph'].mean())
#df.head(5)
#(df['ph']==7).value_counts()


# In[9]:


plt.figure(figsize=(5,5))
df.hist()
plt.show()


# In[10]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


# In[11]:


def water_type(x):
    if x>7:
        y='Basic'
    elif x<7:
        y='Acidic'
    elif x==7:
        y='Neutral'
    return y
        
for x in df['ph']:
        df['pH_Type'] = df['ph'].apply(lambda x: water_type(x))


# In[12]:


df['pH_Type'].value_counts()


# In[13]:


plt.figure(figsize=(7,5))
df['pH_Type'].value_counts().plot(kind='pie',labels = ['','',''], autopct='%1.1f%%', 
                                  colors = ['orange','yellow','salmon'])
plt.legend(labels=['Basic','Acidic','Neutral'])
plt.show()
#sns.countplot(data=df, x='pH_Type',hue='pH_Type')


# In[14]:


def Turbidity_for_potability(q):
    if q>5:
        p='Above range'
    elif q<5:
        p='Medium'
    elif q<=1:
        p='Good for Drinking water'
    return p
df['Turbidity_Type'] = df['Turbidity'].apply(lambda q: Turbidity_for_potability(q))


# In[15]:


df['Turbidity_Type'].value_counts()


# In[16]:


plt.figure(figsize=(7,5))
df['Turbidity_Type'].value_counts().plot(kind='pie',labels = ['','',''], autopct='%1.1f%%', 
                                                   colors = ['yellow','salmon','blue'])

plt.legend(labels=['Above range', 'Medium','Good for Drinking water'])
plt.show()


# In[17]:


df['Turbidity_Type'] = df['Turbidity'].apply(lambda q: Turbidity_for_potability(q))


# In[18]:


def Potability(q):
    if q<1:
        p=1
    else:
        p=0
    return p
df['Potability'] = df['Turbidity'].apply(lambda q: Potability(q)) 
#df['Potability'] = df['pH_Type'].apply(lambda q: Potability(q))


# In[19]:


def Potability_1(q):
    if q<=8.5 and q>=6.5:
        p=1
    else:
        p=0
    return p
df['Potability'] = df['ph'].apply(lambda q: Potability_1(q))


# In[20]:


df.head(5)


# In[21]:


from sklearn.preprocessing import LabelEncoder

pH_Type = ['Neutral', 'Acidic', 'Basic']
Turbidity_Type= ['Good for Drinking water','Medium','Above range']
le = LabelEncoder()
le0 = le.fit(pH_Type)
le1 = le.fit(Turbidity_Type)
df['pH_Type'] = le.fit_transform(df['pH_Type'].astype(str))
df['Turbidity_Type'] = le.fit_transform(df['Turbidity_Type'].astype(str))
df.head(5)

df.to_csv('new_data.csv')


# In[22]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


# In[23]:


X = df.drop('Potability', axis=1)
y = df['Potability']


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=49)
#scale=StandardScaler()
#X_train1=scale.fit_transform(X_train)
#X_test1=scale.transform(X_test)
#y_train=y_train.values
#y_test=y_test.values
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
'''
Accuracy_score=[]
def predict(model):
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    Accuracy_score.append(accuracy_score(y_test,preds))
    print('Accuracy is',accuracy_score(y_test,preds))
    print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
    print('Classification report:',classification_report(y_test,preds))
    joblib.dump(model,'algorithms.sav')


# In[41]:


# XGBClassifier
predict(XGBClassifier())


# In[42]:


# GradientBoostingClassifier
predict(GradientBoostingClassifier())


# In[43]:


# RandomForestClassfier
predict(RandomForestClassifier())


# In[44]:


# DecisionTreeClassifier
predict(DecisionTreeClassifier())


# In[45]:


# SupportVectorClassifier
predict(SVC())
'''
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
preds=gbc.predict(X_test)
Accuracy_score.append(accuracy_score(y_test,preds))
print('Accuracy is',accuracy_score(y_test,preds))
print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
print('Classification report:',classification_report(y_test,preds))
joblib.dump(gbc,'gbclassifier.sav')

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
preds=gbc.predict(X_test)
Accuracy_score.append(accuracy_score(y_test,preds))
print('Accuracy is',accuracy_score(y_test,preds))
print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
print('Classification report:',classification_report(y_test,preds))
joblib.dump(rfc,'rfclassifier.sav')

dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
preds=gbc.predict(X_test)
Accuracy_score.append(accuracy_score(y_test,preds))
print('Accuracy is',accuracy_score(y_test,preds))
print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
print('Classification report:',classification_report(y_test,preds))
joblib.dump(dtc,'dtclassifier.sav')

svc=SVC()
svc.fit(X_train,y_train)
preds=gbc.predict(X_test)
Accuracy_score.append(accuracy_score(y_test,preds))
print('Accuracy is',accuracy_score(y_test,preds))
print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
print('Classification report:',classification_report(y_test,preds))
joblib.dump(svc,'svc.sav')

# In[46]:


Accuracy_score


# In[47]:


# ANN
network=models.Sequential()


# In[48]:


network.add(layers.Dense(units=16, activation='relu',input_shape= (X_train.shape[1],)))
network.add(layers.Dense(units=16, activation='relu'))
network.add(layers.Dense(units=3, activation='sigmoid'))


# In[49]:


network.summary()


# In[50]:


network.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(),
             metrics=['accuracy'])


# In[51]:


y_pred_ann=network.predict(X_test, verbose=0)
y_pred_ann


# In[36]:


#accuracy_score(y_test,abs(y_pred_ann), normalize= False)


# In[39]:


#import joblib
#joblib.dump(model,'algorithms.sav')


# In[ ]:





# In[ ]:





# In[ ]:




