import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler

model = pickle.load(open("model.pkl","rb"))

st.title("Loan Approval Prediction")
st.image("""https://drive.google.com/uc?export=view&id=1ul5FgCwTsmmPxjK5gfJMzwIHnElhGU8R""")
st.header('Enter the Features:')

Gender = st.selectbox('Gender:', ['Male', 'Female'])

Married = st.selectbox('Married:', ['Yes', 'No'])

Dependents = st.selectbox('Number of Dependents:', ['0', '1', '2', '3+'])

Education = st.selectbox('Education:', ['Graduate', 'Not Graduate'])

Self_Employed = st.selectbox('Self Employed:', ['Yes', 'No'])

ApplicantIncome = st.number_input('Applicant Income:', min_value=0)

CoapplicantIncome = st.number_input('Co-applicant Income:', min_value=0.0)

LoanAmount = st.number_input('Loan Amount:', min_value=0.0)

Loan_Amount_Term = st.selectbox('Loan Amount Term:', [360.0, 180.0, 480.0, 300.0, 240.0, 84.0, 120.0, 60.0, 36.0, 12.0])

Credit_History = st.selectbox('Credit History:', [1.0, 0.0])

Property_Area = st.selectbox('Property Area:', ['Semiurban', 'Urban', 'Rural'])

def predict(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):

    if Gender == 'Male':
        Gender = 1
    elif Gender == 'Female':
        Gender = 0

    if Married == 'Yes':
        Married = 1
    elif Married == 'No':
        Married = 0

    if Education == 'Graduate':
        Education = 1
    elif Education == 'Not Graduate':
        Education = 0
 
    if Self_Employed == 'Yes':
        Self_Employed = 1
    elif Self_Employed == 'No':
        Self_Employed = 0
 
    data = pd.DataFrame([[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]], columns=["Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"])

    data["Dependents_0"] = data["Dependents"].apply(lambda x: 1 if x=="0" else 0)
    data["Dependents_1"] = data["Dependents"].apply(lambda x: 1 if x=="1" else 0)
    data["Dependents_2"] = data["Dependents"].apply(lambda x: 1 if x=="2" else 0)
    data["Dependents_3+"] = data["Dependents"].apply(lambda x: 1 if x=="3+" else 0)
    data["Property_Area_Rural"] = data["Property_Area"].apply(lambda x: 1 if x=="Rural" else 0)
    data["Property_Area_Semiurban"] = data["Property_Area"].apply(lambda x: 1 if x=="Semiurban" else 0)
    data["Property_Area_Urban"] = data["Property_Area"].apply(lambda x: 1 if x=="Urban" else 0)

    data = data.drop(['Property_Area','Dependents'], axis = 1)

    def outliers(df1,col):
        IQR=df1[col].quantile(0.75)-df1[col].quantile(0.25)
        LW=df1[col].quantile(0.25)-(IQR*1.5)
        UW=df1[col].quantile(0.75)+(IQR*1.5)
        df1[col]=np.where(df1[col]<LW,df1[col].mean(),df1[col])
        df1[col]=np.where(df1[col]>UW,df1[col].mean(),df1[col])

    num_col = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

    for b in num_col:
        outliers(data,b)

    data.ApplicantIncome = np.sqrt(data.ApplicantIncome)
    data.CoapplicantIncome = np.sqrt(data.CoapplicantIncome)
    data.LoanAmount = np.sqrt(data.LoanAmount)

    data = MinMaxScaler().fit_transform(data)

    prediction = model.predict(data)
    return prediction

if st.button('Predict'):
    Results = predict(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area)
    if Results[0] == 0:
        st.success("We are sorry that your Loan has not been Approved")
    elif Results[0] == 1:
        st.success("Good Luck!, Your Loan has been approved")