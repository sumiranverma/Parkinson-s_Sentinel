# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 03:21:35 2023

@author: LENEVO
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
parkinsons_data = pd.read_csv('C:/Users/LENEVO/Downloads/parkinsons.csv')
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import svm
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)
#loading the model
loaded_model = pickle.load(open('C:/Users/LENEVO/Desktop/ML/trained_model1.sav', 'rb'))

def parkinsons_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the data
    std_data = scaler.transform(input_data_reshaped)

    prediction = model.predict(std_data)
    print(prediction)
    if (prediction[0] == 0):
      return "The Person is not likely to have Parkinson's."

    else:
      return "The Person is likely to have Parkinson's."
def main():
    st.title("Parkinson's Sentinel" )
    MDVPFoHz=st.text_input('MDVP:Fo(Hz)')
    MDVPFhiHz=st.text_input('MDVP:Fhi(Hz)')
    MDVPFloHz=st.text_input('MDVP:Flo(Hz)')
    MDVPJitterpercent=st.text_input('MDVP:Jitter(%)')
    MDVPJitterAbs=st.text_input('MDVP:Jitter(Abs)')
    MDVPRAP=st.text_input('MDVP:RAP')
    MDVPPPQ=st.text_input('MDVP:PPQ ')
    JitterDDP=st.text_input('Jitter:DDP')
    MDVPShimmer=st.text_input('MDVP:Shimmer')
    MDVPShimmerdB=st.text_input(' MDVP:Shimmer(dB)')
    ShimmerAPQ3=st.text_input('Shimmer:APQ3')
    MDVPAPQ=st.text_input('MDVP:APQ')
    ShimmerDDA =st.text_input('Shimmer:DDA')
    NHR=st.text_input('NHR')
    HNR =st.text_input('HNR')
    status=st.text_input('status')
    RPDE=st.text_input('RPDE')
    DFA=st.text_input('DFA')
    spread1=st.text_input('spread1')
    spread2=st.text_input('spread2')
    D2=st.text_input('D2')
    PPE=st.text_input('PPE')
    
    parkinsons=''
    if st.button("Predict Parkinson's"):
        parkinsons=parkinsons_prediction([MDVPFoHz,MDVPFhiHz,MDVPFloHz,MDVPJitterpercent,MDVPJitterAbs,MDVPRAP,MDVPPPQ,JitterDDP,MDVPShimmer,MDVPShimmerdB,ShimmerAPQ3,MDVPAPQ,ShimmerDDA,NHR,HNR,status,RPDE,DFA,spread1,spread2,D2,PPE])
    st.success(parkinsons)    
    
if __name__=='__main__':
    main()
        
    
    
    
              
    
