import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from streamlit_extras.switch_page_button import switch_page





# from streamlit_option_menu import option_menu


# with st.sidebar:
#     selected = option_menu(
#                 menu_title='Main Menu',
#                 options = ['Home','Predict'],
#                 icons = ['house','book'],
#                 menu_icon='cast',
#                 default_index = 0,
#         )

## Setting Page Title
st.set_page_config(initial_sidebar_state="collapsed",
                page_title='Heart Disease Risk Prediction'
                )



## Remove the contents in the sidebar itself
no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)


## Hide the github icon on the right side in the deployed app
hide_github_icon = """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
    </style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)


## To remove the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


## Creating a Container for multiple elements
title = st.container()
abstract = st.expander(label='Abstract',expanded=False)
predict = st.container()


## Writing text in the website

with title:
    st.title('Heart Disease Risk Prediction using Machine Learning Algorithms')
    # st.text()

with abstract:
    st.markdown('<div style="text-align: justify;">For a long time, Cardiovascular diseases (CVD) is still one of the leading cause of \
    death globally. The rise of new technologies such as Machine Learning (ML) algorithms\
    can help with the early detection and prevention of developing CVDs. This study mainly \
    focuses on the utilization of different ML models to determine the risk of a person in \
    developing CVDs by using their personal lifestyle factors. This study used, extracted, and\
    processed the 438,693 records as data from the Behavioral Risk Factor Surveillance System\
    (BRFSS) in 2021 from World Health Organization (WHO). The data was then partitioned\
    into training and testing data with a ratio of 0.8:0.2 to have an unknown data to evaluate\
    the model that will be trained on. One problem that this study faced is the Imbalance\
    among the classes and this was solved by using sampling techniques in order to balance the\
    data for the ML model to process and understand well. The performance of the ML models\
    was evaluated using 10-Stratified Fold cross-validation testing and the best model is Logistic\
    Regression (LR) with F1 score of 0.32564. Logistic Regression model was then subjected to\
    hyperparameter tuning and got the best score of 0.3257 with C = 0.1. Feature Importance\
    was also generated from the LR model and the features that have the most impact is Sex,\
    Diabetes, and the General Health of an individual. After getting the final LR model, it was\
    then evaluated in the testing data and got a F1 score of 0.33. The Confusion Matrix was\
    also used to better visualize the performance. And, the LR model correctly classified 79.18 %\
    of people with CVDs and 73.46 % of people that is healthy. The AUC-ROC Curve was also\
    used as a performance metric and the LR model got an AUC score of 0.837. The Logistic\
    Regression model can be used in the medical field and can be utilized more by adding medical\
    attributes to the data. Overall, this study gave us an insight and significant knowledge that\
    can help in predicting the risk of CVDs by only using the personal attributes of an individual.\
    </div>', unsafe_allow_html=True)

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

def callback():
    st.session_state.button_clicked = True

def go_back():
    st.session_state.button_clicked = False


with predict:
    if st.button('Begin Risk Assessment',on_click=callback) or st.session_state.button_clicked:
        st.write('**Medical Disclaimer:**')
        st.markdown('<div style="text-align: justify;">The contents of this website are not \
        intended to diagnose or treat any disease, or offer personal medical advice. \
        You should seek the advice of your physician or other qualified health provider\
        with any questions you may have regarding a medical condition. Never disregard \
        professional medical advice or delay in seeking it because of something you have\
        read on this website. </div>', unsafe_allow_html=True)
        st.write('')
        if st.button('Agree'):
            st.session_state.button_clicked = False
            switch_page("page_2")
        st.button('Disgree',on_click=go_back)

# st.write(st.session_state)

# with predict:
#     col_1,col_2 = st.columns(2)
#     with col_1:
#         if st.button('Begin Risk Assessment',on_click=callback) or st.session_state.button_clicked:
#             st.write('**Medical Disclaimer:**')
#             st.markdown('<div style="text-align: justify;">The contents of this website are not \
#             intended to diagnose or treat any disease, or offer personal medical advice. \
#             You should seek the advice of your physician or other qualified health provider\
#             with any questions you may have regarding a medical condition. Never disregard \
#             professional medical advice or delay in seeking it because of something you have\
#             read on this website. </div>', unsafe_allow_html=True)
#             st.write('')
#             if st.button('Agree'):
#                 switch_page("page_2")
#             st.button('Disgree',on_click=go_back)



# with predict:
#     col_1,col_2 = st.columns(2)
#     with col_1:
#         if st.button('Begin Risk Assessment'):
#             st.write('**Medical Disclaimer:**')
#             st.markdown('<div style="text-align: justify;">The contents of this website are not \
#             intended to diagnose or treat any disease, or offer personal medical advice. \
#             You should seek the advice of your physician or other qualified health provider\
#             with any questions you may have regarding a medical condition. Never disregard \
#             professional medical advice or delay in seeking it because of something you have\
#             read on this website. </div>', unsafe_allow_html=True)
#             st.write('')
#             if st.button('Agree'):
#                 st.write('hi')
#                 with st.form(key='submit this'):
#                     number_input = st.number_input('Enter a number')
#                     select_box = st.selectbox('Select an option',('1','2','3'))
#                     submit_button = st.form_submit_button(label='Submit')

#             st.button('Disgree')

#     with col_2:
#         if st.button('Next Page'):
#             switch_page("page_2")









