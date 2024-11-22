import streamlit as st
import pickle
import numpy as np

# import the model

pipe = pickle.load(open('pipe_rf.pkl','rb'))
df= pickle.load(open('df.pkl','rb'))


st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df["Company"].unique()) # type: ignore

# type of laptop
type = st.selectbox('Type',df["TypeName"].unique()) # type: ignore

# Ram
ram= st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# touchscreen
is_touchscreen = st.selectbox('touchscreen',['N0','Yes'])

# ips
is_ips = st.selectbox('IPS',['N0','Yes'])

# screensize
screensize = st.number_input("Screen Size")

# Resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768',
                                               '1600x900','3840x2160','3200x1800','2880x1800',
                                               '2560x1600', '2560x1440','2384x1440'])

# cpu
cpu = st.selectbox('CPU',df["Cpu brand"].unique())  # type: ignore

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1084,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1084])

gpu = st.selectbox('GPU',df["Gpu brand"].unique()) # type: ignore

os = st.selectbox('OS',df["os"].unique()) # type: ignore

if st.button("Predict Price"):
    # query
    
    #touchscreen
    if is_touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen= 0
    
    #ips   
    if is_ips== "Yes":
        ips = 1
    else:
        ips= 0
        
    # ppi
    x_res = int(resolution.split("x")[0])
    y_res = int(resolution.split("x")[1])
    ppi =((x_res**2)+(y_res**2))**0.5/screensize
    
    query = np.array([company,type, ram, weight, touchscreen,ips, ppi, cpu, hdd, ssd, gpu ,os])
    
    query = query.reshape(1,12)
    res = str(int(np.exp(pipe.predict(query)[0])))
    st.title("The predicted price of this configuration is: "+ res)