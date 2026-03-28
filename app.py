import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- CONFIG ---
st.set_page_config(page_title="StyleNear Dashboard", layout="wide")

# --- DATA GENERATOR ---
@st.cache_data
def get_data():
    np.random.seed(42)
    n = 1000 # Reduced size to save memory on free tier
    regions = ['North', 'South', 'East', 'West', 'Central']
    categories = ['Kurtas', 'Sarees', 'Western Tops', 'Denim', 'Ethnic Sets']
    
    df = pd.DataFrame({
        'Age': np.random.randint(18, 60, n),
        'Region': np.random.choice(regions, n),
        'Income_Group': np.random.choice(['<25k', '25k-50k', '50k-1L', '>1L'], n),
        'Category': np.random.choice(categories, n),
        'Monthly_Budget': np.random.randint(500, 8000, n),
        'Size_Confidence': np.random.randint(1, 11, n)
    })
    # Target Logic
    df['Purchase_Intent'] = ((df['Monthly_Budget'] > 3000) & (df['Size_Confidence'] > 5)).astype(int)
    return df

# --- MAIN APP ---
st.title("🚀 StyleNear Founder Dashboard")
df = get_data()

tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "👥 Segments", "🔮 Prediction"])

with tab1:
    st.subheader("Regional Demand")
    fig = px.histogram(df, x="Region", color="Category", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Customer Segmentation")
    try:
        X_km = df[['Age', 'Monthly_Budget', 'Size_Confidence']]
        # Added n_init="auto" to be compatible with latest scikit-learn
        km = KMeans(n_clusters=3, n_init="auto", random_state=42).fit(X_km)
        df['Cluster'] = km.labels_
        fig2 = px.scatter(df, x="Monthly_Budget", y="Age", color=df['Cluster'].astype(str))
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Clustering Error: {e}")

with tab3:
    st.subheader("Purchase Prediction")
    try:
        # Simple One-Hot Encoding
        X = pd.get_dummies(df[['Age', 'Region', 'Monthly_Budget', 'Size_Confidence']])
        y = df['Purchase_Intent']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
        c3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
        
        st.write("### Predict Future Lead")
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            st.success("File Received! Model ready for scoring.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
