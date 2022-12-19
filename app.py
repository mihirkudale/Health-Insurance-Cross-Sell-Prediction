import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from PIL import Image
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import plotly.express as px
import warnings 
warnings.filterwarnings("ignore")

#@app.route('/')
def welcome():
    return "Welcome All"


st.write("""
# Cross-sell Prediction ML App

This app predicts if the existing health insurance customers will buy vehicle insurance!
Data obtained from the [kaggle](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction).
""")

img = Image.open("health-insurance.jpg")
 
# display image using streamlit
# width is used to set the width of an image
st.image(img, width=704)

hide_menu_style = """
          <style>
          footer {visibility: hidden;}
          </style>   
"""
st.markdown(hide_menu_style, unsafe_allow_html= True)

st.sidebar.header('Customer Details')
html_temp = """
 <div style="background-color:tomato;padding:10px">
 <h2 style="color:white;text-align:center;">Cross-Sell Predictor</h2>
 </div>
 """
st.markdown(html_temp,unsafe_allow_html=True)

def customer_details():

    gender = st.sidebar.radio('Gender', ('Male', 'Female'))
    if gender == 'Male':
        st.sidebar.write("**Gender:**",'Male')
    else:
        st.sidebar.write("**Gender:**",'Female')

    age = st.sidebar.number_input('Age (years)', 20, 85, 52)
    st.sidebar.write("**Your age is:**", age)

    dl = st.sidebar.selectbox('Have driving license?', ('Yes', 'No'))
    if dl == 'Yes':
        st.sidebar.write("**Have driving license:**",'Yes')
    else:
        st.sidebar.write("**Have driving license:**",'No')

    region_code = st.sidebar.number_input('Region code', 0, 52, 25)
    st.sidebar.write("**Your region code is:**", region_code)

    previously_insured = st.sidebar.selectbox(
        'Previously insured?', ('Yes', 'No'))
    if previously_insured == 'Yes':
        st.sidebar.write("**Previously insured:**",'Yes')
    else:
        st.sidebar.write("**Previously insured:**",'No')

    vehicle_age = st.sidebar.radio(
        'Vehicle Age', ('< 1 Year', '1-2 Year', '> 2 Years'))
    st.sidebar.write("**Your vehicle age is:**", vehicle_age)

    vehicle_damage = st.sidebar.selectbox('Vehicle Damage', ('Yes', 'No'))
    if vehicle_damage == 'Yes':
        st.sidebar.write("**Vehicle Damaged:**",'Yes')
    else:
        st.sidebar.write("**Vehicle Damaged:**",'No')
   
    annual_premium = st.sidebar.number_input('Annual Premium', 2630, 540165, 271397)
    st.sidebar.write("**Annual Premium:**", annual_premium)
    
    policy_sales_channel = st.sidebar.number_input(
        'Policy Sales Channel', 1, 163, 80)
    st.sidebar.write("**Policy Sales Channel:**", policy_sales_channel)
    
    vintage = st.sidebar.number_input('Vintage', 10, 299, 150)
    st.sidebar.write("**Vintage:**", vintage)

    def change_text_to_num(x):
        if x == "Yes":
            return 1
        return 0

    vehicle_age_dict={'< 1 Year':1, '1-2 Year':2, '> 2 Years':3}
    gender_dict= {'Male':1,'Female':0}
    vehicle_damage_dict= {'Yes':1,"No":0}
    dl_dict= {'Yes':1,'No':0}
    
    data = {'Gender': gender_dict[gender],
            'Age': age,
            'Driving_License': dl_dict[dl],
            'Region_Code': region_code,
            'Previously_Insured': change_text_to_num(previously_insured),
            'Vehicle_Age': vehicle_age_dict[vehicle_age],
            'Vehicle_Damage': vehicle_damage_dict[vehicle_damage],
            'Annual_Premium': annual_premium,
            'Policy_Sales_Channel': policy_sales_channel,
            'Vintage': vintage}

    features = pd.DataFrame(data, index=[0])
    return features


input_df = customer_details()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
raw_data = pd.read_csv('test.csv')
insurance = raw_data.drop(columns=['id'])

# # # Encoding of ordinal features


def preprocessor(x):
    encode = ["Gender", 'Vehicle_Age', 'Vehicle_Damage', 'Driving_License']

    for col in encode:
        dummy = pd.get_dummies(x[col], prefix=col)
        x = pd.concat([x, dummy], axis=1)
        del x[col]

    return x


numer = ['Age', 'Annual_Premium',
         'Policy_Sales_Channel', 'Vintage', 'Region_Code']

# Displays the customer details
st.subheader('Customer Details')

st.subheader("This is default dataframe")

with st.spinner('Loading...'):
    time.sleep(1)

st.write(insurance)

shape = insurance.shape
st.write('Number of rows :', shape[0])
st.write('Number of columns :', shape[1])

st.subheader("This is AgGrid Table")

with st.spinner('Loading...'):
    time.sleep(4)

def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="material",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection

selection = aggrid_interactive_table(df=insurance)

if selection:
    st.write("You selected:")
    st.json(selection["selected_rows"])

# Reads in saved classification model
pickle_in= open("model_tuned.pkl","rb")
model=pickle.load(pickle_in)

# Apply model to make predictions
prediction = model.predict(np.array(input_df))
prediction_proba = model.predict_proba(np.array(input_df))

# Reads in saved random forest model
pick= open("model_3.pkl","rb")
model3=pickle.load(pick)

# Apply model to make predictions
prediction3 = model3.predict(np.array(input_df))
prediction_proba3 = model3.predict_proba(np.array(input_df))

pred = ['No', 'Yes']

st.subheader('Will the customer buy?')
st.markdown('**DT Tuned model:-**')
st.write(pred[int(prediction)])

st.markdown('**RF Tuned model:-** ')
st.write(pred[int(prediction3)])

st.subheader('Prediction Probability')
st.write('**DT-**')
predict1 = pd.DataFrame(
    prediction_proba,
     columns=["Not Interested", "Interested"]
     )
st.write(predict1)
st.bar_chart(predict1)

st.write('**RF-**')
predict2 = pd.DataFrame(
    prediction_proba3,
     columns=["Not Interested", "Interested"]
     )
st.write(predict2)
st.bar_chart(predict2)
