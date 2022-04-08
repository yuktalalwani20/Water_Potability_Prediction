import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly.express as px
import joblib
import base64


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





df= pd.read_csv('water_potability_2.csv')
#df.head(10)
#df.info()
#df.isnull().sum()
#df.describe()
df=df.dropna()
#df.head(10)


#df['ph']=df['ph'].fillna(df['ph'].mean())
#df.head(5)
#(df['ph']==7).value_counts()


#plt.figure(figsize=(5,5))
#df.hist()
#plt.show()

#plt.figure(figsize=(10,8))
#sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

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

#df['pH_Type'].value_counts()


#plt.figure(figsize=(7,5))
#df['pH_Type'].value_counts().plot(kind='pie',labels = ['','',''], autopct='%1.1f%%', colors = ['orange','yellow','salmon'])
#plt.legend(labels=['Basic','Acidic','Neutral'])
#plt.show()
#sns.countplot(data=df, x='pH_Type',hue='pH_Type')


def Turbidity_for_potability(q):
    if q>5:
        p='Above range'
    elif q<5:
        p='Medium'
    elif q<=1:
        p='Good for Drinking water'
    return p
df['Turbidity_Type'] = df['Turbidity'].apply(lambda q: Turbidity_for_potability(q))


#df['Turbidity_Type'].value_counts()


#plt.figure(figsize=(7,5))
#df['Turbidity_Type'].value_counts().plot(kind='pie',labels = ['','',''], autopct='%1.1f%%', colors = ['yellow','salmon','blue'])

#plt.legend(labels=['Above range', 'Medium','Good for Drinking water'])
#plt.show()


#df['Turbidity_Type'] = df['Turbidity'].apply(lambda q: Turbidity_for_potability(q))


# In[18]:


def Potability(q):
    if q<1:
        p=1
    else:
        p=0
    return p
df['Potability'] = df['Turbidity'].apply(lambda q: Potability(q)) 
#df['Potability'] = df['pH_Type'].apply(lambda q: Potability(q))


def Potability_1(q):
    if q<=8.5 and q>=6.5:
        p=1
    else:
        p=0
    return p
df['Potability'] = df['ph'].apply(lambda q: Potability_1(q))


#df.head(5)


from sklearn.preprocessing import LabelEncoder

pH_Type = ['Neutral', 'Acidic', 'Basic']
Turbidity_Type= ['Good for Drinking water','Medium','Above range']
le = LabelEncoder()
le0 = le.fit(pH_Type)
le1 = le.fit(Turbidity_Type)
df['pH_Type'] = le.fit_transform(df['pH_Type'].astype(str))
df['Turbidity_Type'] = le.fit_transform(df['Turbidity_Type'].astype(str))
#df.head(5)
df.to_csv('new_data.csv')

#plt.figure(figsize=(10,8))
#sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


X = df.drop('Potability', axis=1)
y = df['Potability']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=49)
#scale=StandardScaler()
#X_train1=scale.fit_transform(X_train)
#X_test1=scale.transform(X_test)
#y_train=y_train.values
#y_test=y_test.values
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

Accuracy_score=[]
#def predict(model):
 #   model.fit(X_train,y_train)
  #  preds=model.predict(X_test)
   # Accuracy_score.append(accuracy_score(y_test,preds))
   # print('Accuracy is',accuracy_score(y_test,preds))
    #print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
    #print('Classification report:',classification_report(y_test,preds))
    #joblib.dump(model,'algorithms.sav')



# XGBClassifier
#predict(XGBClassifier())
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
preds=xgb.predict(X_test)
Accuracy_score.append(accuracy_score(y_test,preds))
#print('Accuracy is',accuracy_score(y_test,preds))
#print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
#print('Classification report:',classification_report(y_test,preds))
joblib.dump(xgb,'xgbclassifier.sav')


# GradientBoostingClassifier
#predict(GradientBoostingClassifier())
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
preds=gbc.predict(X_test)
Accuracy_score.append(accuracy_score(y_test,preds))
#print('Accuracy is',accuracy_score(y_test,preds))
#print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
#print('Classification report:',classification_report(y_test,preds))
joblib.dump(gbc,'gbclassifier.sav')


# RandomForestClassfier
#predict(RandomForestClassifier())
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
preds=rfc.predict(X_test)
Accuracy_score.append(accuracy_score(y_test,preds))
#print('Accuracy is',accuracy_score(y_test,preds))
#print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
#print('Classification report:',classification_report(y_test,preds))
joblib.dump(rfc,'rfclassifier.sav')


# DecisionTreeClassifier
#predict(DecisionTreeClassifier())
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
preds=dtc.predict(X_test)
Accuracy_score.append(accuracy_score(y_test,preds))
#print('Accuracy is',accuracy_score(y_test,preds))
#print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
#print('Classification report:',classification_report(y_test,preds))
joblib.dump(dtc,'dtclassifier.sav')


# SupportVectorClassifier
#predict(SVC())
svc=SVC()
svc.fit(X_train,y_train)
preds=svc.predict(X_test)
Accuracy_score.append(accuracy_score(y_test,preds))
#print('Accuracy is',accuracy_score(y_test,preds))
#print('Confusion matrix of the model is',confusion_matrix(y_test,preds))
#print('Classification report:',classification_report(y_test,preds))
joblib.dump(svc,'svc.sav')


#Accuracy_score



# ANN
network=models.Sequential()

network.add(layers.Dense(units=16, activation='relu',input_shape= (X_train.shape[1],)))
network.add(layers.Dense(units=16, activation='relu'))
network.add(layers.Dense(units=3, activation='sigmoid'))

network.summary()

network.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(),
             metrics=['accuracy'])


y_pred_ann=network.predict(X_test, verbose=0)
#y_pred_ann





data = pd.read_csv('new_data.csv')
data = data.iloc[:,0:]
original_data = pd.read_csv('new_data.csv')
feature_list = {'ph':0,'Turbidity':0,'pH_Type':0,'Turbidity_Type':0}

classifier = joblib.load('algorithms.sav')
rfc=joblib.load('rfclassifier.sav')
dtc=joblib.load('dtclassifier.sav')
xgbc=joblib.load('xgbclassifier.sav')
gbc=joblib.load('gbclassifier.sav')
svc=joblib.load('svc.sav')
ann=joblib.load('ann.sav')
main_bg = "nature-3267579_1920.jpg"
main_bg_ext = "jpg"

side_bg = "nature-3267579_1920.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Water Potability Prediction !!!!!!!')
st.subheader('Predict the water you drink is pure or not ??')
st.sidebar.header('Predict The Purity')
#for j in feature_list.keys():
 #   feature_list[j] = st.sidebar.text_input(f'enter value for {j}')
avg0=df['ph'].mean()
avg1=df['Turbidity'].mean()
avg2=max(df['pH_Type'])
avg3=max(df['Turbidity_Type'])
st.sidebar.write('pH',avg0)
st.sidebar.write('Turbidity',avg1)
st.sidebar.write('pH Type',avg2)
st.sidebar.write('Turbidity Type',avg3)


'''

if st.sidebar.button('Predict_gb'):
    if 0 in list(feature_list.values()):
        st.sidebar.markdown(' # Please fill all the values')
    else:
        #if st.sidebar.button('RF'):
        pred = gbc.predict([list(feature_list.values())])
        if pred[0]==0:
            st.sidebar.markdown('# water is not so potable for drinking purpose')
        else:
            st.sidebar.markdown('# water is potable for drinking purpose')
          
if st.sidebar.button('Predict_sv'):
    if 0 in list(feature_list.values()):
        st.sidebar.markdown(' # Please fill all the values')
    else:
        #if st.sidebar.button('RF'):
        pred = svc.predict([list(feature_list.values())])
        if pred[0]==0:
            st.sidebar.markdown('# water is not so potable for drinking purpose')
        else:
            st.sidebar.markdown('# water is potable for drinking purpose')    
if st.sidebar.button('Predict_rf'):
    if 0 in list(feature_list.values()):
        st.sidebar.markdown(' # Please fill all the values')
    else:
        #if st.sidebar.button('RF'):
        pred = rfc.predict([list(feature_list.values())])
        if pred[0]==0:
            st.sidebar.markdown('# water is not so potable for drinking purpose')
        else:
            st.sidebar.markdown('# water is potable for drinking purpose')
if st.sidebar.button('Predict_dt'):
    if 0 in list(feature_list.values()):
        st.sidebar.markdown(' # Please fill all the values')
    else:
        #if st.sidebar.button('RF'):
        pred = dtc.predict([list(feature_list.values())])
        if pred[0]==0:
            st.sidebar.markdown('# water is not so potable for drinking purpose')
        else:
            st.sidebar.markdown('# water is potable for drinking purpose')

if st.sidebar.button('Predict_xgb'):
    if 0 in list(feature_list.values()):
        st.sidebar.markdown(' # Please fill all the values')
    else:
        #if st.sidebar.button('RF'):
        pred = xgbc.predict([np.array(feature_list.values())])
        if pred[0]==0:
            st.sidebar.markdown('# water is not so potable for drinking purpose')
        else:
            st.sidebar.markdown('# water is potable for drinking purpose')

if st.sidebar.button('Predict_ann'):
    if 0 in list(feature_list.values()):
        st.sidebar.markdown(' # Please fill all the values')
    else:
        #if st.sidebar.button('RF'):
        pred = ann.predict([list(feature_list.values())])
        if pred[0]==0:
            st.sidebar.markdown('# water is not so potable for drinking purpose')
        else:
            st.sidebar.markdown('# water is potable for drinking purpose')
  '''       
st.image('water.jpg')
st.dataframe(data)
header = st.container()
body = st.container()
with header:
    col1,col2 = st.columns(2)
    plot_type = col1.selectbox('Plot the feature',['histogram','line plot','area chart'])
    feat = col1.selectbox('Which feature', ['ph','Turbidity'])
    if plot_type == 'histogram':
        col2.bar_chart(data[feat][:200])
        col1.header(f'{plot_type} of {feat} feature')
    if plot_type == 'line plot':
        col2.line_chart(data[feat][:200])
        col1.header(f'{plot_type} of {feat} feature')
    if plot_type == 'area chart':
        col2.area_chart(data[feat][:200])
        col1.header(f'{plot_type} of {feat} feature')
with body:
    col1, col2 = st.columns(2)
    col1.header('About Me')
    col1.markdown('''My name is Yukta Lalwani , I'm pursuing my Computer Science Degree and love to do Machine Learning stuff''')
    col2.header('About Project')
    col2.markdown("Are you researching for water potability check, Just fill the content of water and here you go")
