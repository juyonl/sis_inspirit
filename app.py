import streamlit as st
from joblib import load
import numpy as np 
import os
import sklearn

# Imports the functions we coded above
from header import *
from userinput import *
from response import *
from predictor import *

# Load our DecisionTree model into our web app
print(numpy.__version__)
print(joblib.__version__)
print(sklearn.__version__)
model = load("model.joblib")
st.write ("Model uploaded!") # You may remove this in your finalized web app!

create_header()
input_features = get_user_input()
prediction = make_prediction(model, input_features)
get_app_response(prediction)
