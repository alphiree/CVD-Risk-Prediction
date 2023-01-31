import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from streamlit_extras.switch_page_button import switch_page

# no_sidebar_style = """
#     <style>
#         div[data-testid="stSidebarNav"] {display: none;}
#     </style>
# """
# st.markdown(no_sidebar_style, unsafe_allow_html=True)

# st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)



## Setting Page Title
st.set_page_config(initial_sidebar_state="collapsed",
                page_title='Heart Disease Risk Prediction'
                )
                
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def model():
    final_model = joblib.load('Results/final_model')
    return final_model

final_model = model()

## Creating a Container for multiple elements
title = st.container()
predict = st.container()


## Writing text in the website

with title:
    st.title('Heart Disease Risk Prediction using Machine Learning Algorithms')
    st.write('By: Ryan Marcus Jeremy M. Lupague')
    # st.text()
    if st.button('Main Page'):
        switch_page('webapp')
    # st.subheader('Abstract:')
    # st.text()

# with predict:
#     col_1,col_2 = st.columns(2)
#     with col_1:
#         with st.form(key='submit this'):
#             st.subheader('Fill out the Following:')


#             name = st.text_input('Enter your name:')
#             number_input = st.number_input('Enter a number')
#             select_box = st.selectbox('Select an option',('1','2','3'))
#             submit_button = st.form_submit_button(label='Submit')

with predict:

    with st.form(key='submit this'):
        st.subheader('Fill out the Following:')
        name = st.text_input('Enter your name:')

        st.write('**Demographic and Screening Questions**')
        Age_Category = st.selectbox('In what Age category do you belong?',('Select One','18-24',
                                                                            '25-29',
                                                                            '30-34',
                                                                            '35-39',
                                                                            '40-44',
                                                                            '45-49',
                                                                            '50-54',
                                                                            '55-59',
                                                                            '60-64',
                                                                            '65-69',
                                                                            '70-74',
                                                                            '75-79',
                                                                            '80+',))
                                                    
        Sex = st.selectbox('Sex',('Select One','Male','Female'))


        Height_cm = st.slider(
                                'Height (cm)',
                                25, 300,step=1)
        
        Weight_kg = st.slider(
                                'Weight (kg)',
                                25, 300,step=1)
                                
        Smoking_History = st.radio('Have you smoked at least 100 cigarettes in your entire life?',
                                    ('No','Yes'),horizontal=True)

        st.write('**Health Status**')
        General_Health = st.selectbox('Would you say that in general, your health is',('Select One',
                                                                            'Poor',
                                                                            'Fair',
                                                                            'Good',
                                                                            'Very Good',
                                                                            'Excellent',
                                                                                    ))
        st.write('**Health Care Access**')
        Checkup = st.selectbox('About how long has it been since you \
                                    last visited a doctor for a routine checkup?',('Select One',
                                                                            'Within the past year',
                                                                            'Within the past 2 years',
                                                                            'Within the past 5 years',
                                                                            '5 or more years ago',
                                                                            'Never',
                                                                                    ))

        st.write('**Exercise**')
        Exercise = st.radio('During the past month, other than your regular job, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking for exercise?',
                                    ('Yes','No'),horizontal=True)

        st.write('**Health Conditions**')
        st.write('Has a doctor, nurse, or other health professional ever told you that you had any of the following? For each, tell me Yes or No.')
        Depression = st.radio('(Ever told) (you had) a depressive disorder (including depression, major depression, dysthymia, or minor depression)?',
                                    ('No','Yes'),horizontal=True)
        Diabetes = st.radio('(Ever told) (you had) diabetes?',
                                    ('No','Yes'),horizontal=True)
        Arthritis = st.radio('(Ever told) (you had) some form of arthritis, rheumatoid arthritis, gout, lupus, or fibromyalgia?',
                                    ('No','Yes'),horizontal=True)
        Skin_Cancer = st.radio('(Ever told) (you had) skin cancer?',
                                    ('No','Yes'),horizontal=True)
        Other_Cancer = st.radio('(Ever told) (you had) any other types of cancer?',
                                    ('No','Yes'),horizontal=True)
        
        st.write('**Food and Drink Consumption**')

        Alcohol_Consumption = st.slider(
                                'During the past 30 days, how many days \
                                did you have at least one drink of any alcoholic beverage such \
                                as beer, wine, a malt beverage or liquor?',
                                0, 30,step=1)
        
        st.write('Now think about the foods you ate during the past month, that is, the past 30 days, including meals and snacks.')
        Fruit_Consumption = st.slider(
                                'Not including juices, how often did you eat fruit?',
                                0, 120,step=1)
        Green_Vegetables_Consumption = st.slider(
                                'How often did you eat a green leafy or lettuce salad, with or without other vegetables?',
                                0, 120,step=1)
        FriedPotato_Consumption = st.slider(
                                'How often did you eat any kind of fried potatoes, including French fries, home fries, or hash browns?',
                                0, 120,step=1)
        

        submit_button = st.form_submit_button(label='Predict')

bmi = Weight_kg / (Height_cm/100)**2

new_input = [General_Health,Checkup,Exercise,Skin_Cancer,
            Other_Cancer,Depression,Diabetes,Arthritis,
            Sex,Age_Category,Height_cm,Weight_kg,bmi,
            Smoking_History,Alcohol_Consumption,Fruit_Consumption,
            Green_Vegetables_Consumption,FriedPotato_Consumption
]

# new_input_2 = ['Good',
#  'Within the past year',
#  'Yes',
#  'Yes',
#  'No',
#  'No',
#  'No',
#  'No',
#  'Male',
#  '70-74',
#  180.0,
#  92.53,
#  28.45,
#  'Yes',
#  12.0,
#  4.0,
#  30.0,
#  0.0]


df = pd.DataFrame([new_input])

df.columns = ['General_Health',
 'Checkup',
 'Exercise',
 'Skin_Cancer',
 'Other_Cancer',
 'Depression',
 'Diabetes',
 'Arthritis',
 'Sex',
 'Age_Category',
 'Height_(cm)',
 'Weight_(kg)',
 'BMI',
 'Smoking_History',
 'Alcohol_Consumption',
 'Fruit_Consumption',
 'Green_Vegetables_Consumption',
 'FriedPotato_Consumption']


st.subheader('Results')
if submit_button:
    try:
        pred = final_model.predict(df)
        y_pred_proba = final_model.predict_proba(df)
        st.write(f'Hello, {name}!')
        st.write('Based from the Machine Learning model, your risk of developing Cardiovascular Disease (CVD) is:')

        if pred[0] == 0:
            risk = 'LOW'
            st.success(f'**{risk}**')
        else:
            risk = 'HIGH'
            st.error(f'**{risk}**')

        
        st.warning('Disclaimer: **The results from this test are not intended to diagnose or treat any disease, or offer personal medical advice.**\
                The model was only trained in 300,000 data and with personal attributes only. Moreover, the analysis of\
                    this models states that it is likely to classify certain attributes such as the sex of the person, their general health status, and being \
                        diabetic as high importance in determining if the person is at risk or not.' )
        
        st.info('Accuracy: The Machine Learning Model used for prediction was first evaluated in an unknown data consisting of 60,000 records, \
            The model correctly classified 79.18 % of people with CVDs and 73.46 % of people that is healthy. However, only 21 % of the predicted by the model out of all predicted that are at risk is correctly classified. \
                The model takes into consideration the cost of misclassifying at risk people as healthy. That is why the model is more likely to classify people at risk.' )

        details = st.expander(label='More Details',expanded=False)
        with details:
            st.write('According to the ML model:')
            st.write('The Probability that it will classify you as at risk for CVDs are:')
            st.info(y_pred_proba[:,1][0])
            st.write('The Probability that it will classify you as at no risk for CVDs are:')
            st.info(y_pred_proba[:,0][0])
            st.write('Note: If the Probability it will classify you at risk is over 0.5, then the model will classify you as at risk for CVDs')

        
        st.balloons()


        



        
    except:
        st.error('Enter valid values to show the results')
        #pass


