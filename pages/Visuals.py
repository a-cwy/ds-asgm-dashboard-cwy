import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport

from streamlit_pandas_profiling import st_profile_report

df = pd.read_csv('dataset/online_gaming_behavior_dataset.csv')
pr = ProfileReport(df, title="Report")

## PAGE
st.set_page_config(
    page_title='Data Visuals',
    layout='wide'
)

st.title("Data Visualization")
st_profile_report(pr)