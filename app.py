import streamlit as  st 
import pickle
import numpy as np
import pandas as pd


lr1=pickle.load(open,('lr1_model_insurance.pkl','rb'))
dt1=pickle.load(open,('dt1_model_insurance.pkl','rb'))
rf1=pickle.load(open,('rf1_model_insurance.pkl','rb'))


st.title('Insurance charge prediction app')

st.header('Fill the details to generate the predicted insurance charge')

options=st.sidebar.selectbox('select ML model',['Liner_Reg','Decision_tree','Random_Forest'])

# form widgets ( they are inbuilt function of streamlit)
# slider, Drop Down, Input Box

age=st.slider(18,64)
sex=st.selectbox('sex',['Male','Female'])
bmi=st.slider(6,53)
children=st.selectbox('children',[0,1,2,3,4,5])
smoker=st.selectbox('smoker',['yes','No'])
region=st.selectbox('region',['NWest','SEast','SWest','NEast'])


if st.button(predict):
    if sex=='Male':
        sex=1
    else:
        sex=0
    if smoker=='yes':
        smoker=1
    else:
        smoker=0
    if region=='NWest':
        region=1
    elif region=='SEast':
        region=2
    elif region=='SWest':
        region=1
    else:
        region=3
             
    
test=np.array(age,sex,bmi,children,smoker,region)
test=test.reshape(1,6)
if options=='Liner_Reg':
    st.success(lr1.predict(test)[0])
elif option=='Decision_tree':
    st.success(dt1.predict(test)[0])
else:
    st.success(rf1.predict(test)[0])
