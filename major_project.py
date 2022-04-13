import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import joblib
import base64


data = pd.read_csv('new_data.csv')
data = data.iloc[:,0:]
original_data = pd.read_csv('new_data.csv')
#feature_list = {'ph':0,'Turbidity':0,'pH_Type':0,'Turbidity_Type':0}

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
 #   avg=sum(data/data.value_counts())
  #  feature_list[j] = st.write(avg)
#avg0=sum(data['ph'])/len(data['ph'])
#avg1=sum(data['Turbidity'])/len(data['Turbidity'])
#avg2=sum(data['pH_Type'])/len(data['pH_Type'])
#avg3=sum(data['Turbidity_Type'])/len(data['Turbidity_Type'])
#ph=st.write(avg0)
#turbidity=st.write(avg1)
#ph_type=st.write(avg2)
#turbidity_type=st.write(avg3)
avg0=data['ph'].mean()
avg1=data['Turbidity'].mean()
#avg2=data['pH_Type'].mean()
#avg3=data['Turbidity_Type'].mean()
st.sidebar.write(avg0)
st.sidebar.write(avg1)
#st.write(avg2)
#st.write(avg3)


if st.sidebar.button('Predict_rf'):
   # if 0 in list(feature_list.values()):
    #    st.sidebar.markdown(' # Please fill all the values')
    #else:
        #if st.sidebar.button('RF'):
    pred = rfc.predict([[avg0,avg1]])
    if pred==0:
       st.sidebar.markdown('# Water is not so potable for further consumption')
    else:
       st.sidebar.markdown('# Water is potable for further consumption')

 if avg0>7:
    st.write('Water is still basic')
 elif avg0<7:
    st.write('Water is still acidic')
 elif avg0==7:
    st.write('Water is still neutral')

 if avg1>5:
    st.write('Turbidity is above range')
 elif avg1<5:
    st.write('Turbidity is medium')
 elif avg1==7: and avg1>1:
    st.write('Water is still good for drinking')

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
    col1.header('About Us')
    #col1.markdown("My name is Yukta Lalwani , I'm pursuing my Computer Science Degree and love to do Machine Learning stuff")
    col1.markdown("This project is made by Khushi Shahu, Kiran Assudani, Krishan Harwani, Yukta Lalwani")
   # col2.header('About Project')
   # col2.markdown("Are you researching for water potability check, Just fill the content of water and here you go")
    

        
