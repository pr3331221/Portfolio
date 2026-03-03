import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def audit_inventory(df):
    prompt = f"""
    You're an AI trained to analyze inventory for inefficiencies. Please identify:
    - Duplicate SKUs
    - Idle inventory (not used in 12+ months)
    - Overstocked items (more than 24 months supply)
    - Items near warranty expiration (within 90 days)

    Here's the inventory:\n{df.head(50).to_string(index=False)}
    """
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

st.set_page_config(page_title="AI Inventory Auditor", layout="wide")
st.title("AI Spare Parts Auditor")

uploaded_file = st.file_uploader("📂 Upload your inventory CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df)

    if st.button("🔍 Run AI Audit"):
        with st.spinner("Analyzing your data with GPT..."):
            audit = audit_inventory(df)
            st.success("✅ Audit Completed")
            st.text_area("📄 AI Audit Report", audit, height=300)

            st.download_button(
                label="📥 Download Report",
                data=audit,
                file_name="AI_Inventory_Audit.txt",
                mime="text/plain"
            )
import pandas as pd

# Load CSV
df = pd.read_csv("sample.csv")
print("Original Data:")
print(df)

# Example of basic cleaning
df.dropna(inplace=True)  # Drop rows with missing values
df.columns = [col.strip() for col in df.columns]  # Clean column names

print("\nCleaned Data:")
print(df)
import streamlit as st
import pandas as pd

st.title("Spare Parts CSV Viewer")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview:")
    st.dataframe(df)
import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def audit_inventory(df):
    prompt = f"""
    You are an expert auditor. Analyze this inventory and flag:
    - Duplicate parts
    - Idle parts (not used in 12+ months)
    - Excess quantities
    - Warranty risks

    Data:\n{df.head(50).to_string(index=False)}
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content

st.title("AI Inventory Auditor")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Your Inventory Data")
    st.dataframe(df)

    if st.button("Run Audit"):
        with st.spinner("Auditing with AI..."):
            report = audit_inventory(df)
            st.success("Audit complete!")
            st.text_area("Audit Report", report, height=300)
