import os
import streamlit as st
import pandas as pd
import joblib
from glove_transformer import GloveVectorTransformer

path = os.path.dirname(__file__)
model_file = path+'/ltv_predictions_pipeline.pkl'
loaded_pipeline = joblib.load(model_file)

left_column, right_column = st.columns(2, gap="large")
with st.sidebar:
    numeric_features = [
        ('total_amount', 0, 5000),
        ('num_orders', 1, 5),
        ('avg_value', 1, 500),
        ('customer_retention', 0, 100),
        ('avg_lead_time', 0, 100)
        ]
    widgets = [st.slider(k, min_value=low, max_value=high, value=(high-low)//2) for k, low, high in numeric_features]
    country = st.selectbox('Country', 
        ['Austria', 'Cyprus', 'Belgium', 'Australia', 'RSA', 'Denmark',
       'Germany', 'France', 'Norway', 'Unspecified', 'Sweden', 'Spain',
       'USA', 'Italy', 'Finland', 'United Kingdom', 'Japan', 'Portugal',
       'Netherlands', 'Poland', 'United Arab Emirates', 'Switzerland',
       'EIRE', 'Greece', 'Channel Islands', 'Malta', 'Bahrain',
       'Singapore', 'Thailand', 'Israel', 'Lithuania', 'Nigeria',
       'West Indies'])
    description = st.text_area("Description")

data = {
    numeric_features[i][0]: [widgets[i]] for i in range(len(numeric_features))
    } | {
        'country': [country],
        'description': [description]
    } 

df = pd.DataFrame(data)
predicted = loaded_pipeline.predict(df)

st.write(f"# Customer is estimated to spend ${round(predicted[0],2)} in the next one year ")
