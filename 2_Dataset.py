import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset", page_icon="📊", layout="wide")

st.title("📊 Heart Disease Dataset")

df = pd.read_csv("cleaned_heart_disease_data.csv")

st.subheader("Dataset Preview")
st.dataframe(df, use_container_width=True)

st.subheader("Dataset Shape")

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

st.subheader("Column Information")
st.dataframe(df.dtypes.astype(str))

st.subheader("Summary Statistics")
st.dataframe(df.describe())

st.download_button(
    "⬇ Download Dataset",
    data=df.to_csv(index=False),
    file_name="cleaned_heart_disease_data.csv",
    mime="text/csv"
)

