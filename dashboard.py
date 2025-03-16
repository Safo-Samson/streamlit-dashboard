import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./CW_Dataset_4100263.csv")
    return df

# Train the model
@st.cache_data
def train_model(df):
    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, scaler

# Streamlit App
def main():
    st.set_page_config(page_title="Quality Prediction Dashboard", layout="wide")
    
    # Sidebar
    st.sidebar.header("User Input Parameters")
    df = load_data()
    model, X_test, y_test, scaler = train_model(df)

    # User input form for process parameters
    with st.sidebar.expander("🔧 Enter Process Parameters"):
        input_data = {feature: st.number_input(f"{feature}", value=df[feature].mean()) for feature in df.columns[:-1]}

    # Predict quality class
    if st.sidebar.button("🔍 Predict Quality"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        with st.spinner("Predicting..."):  # Show a loading spinner
            time.sleep(2)  # Simulate processing time
            prediction = model.predict(input_scaled)
        
        st.sidebar.success(f"Predicted Quality Class: **{prediction[0]}** 🎯")

    # Dashboard Layout
    st.title("📊 Product Quality Prediction Dashboard")
    st.markdown("Analyze and predict product quality using Machine Learning.")

    # Model Performance Section
    st.subheader("🏆 Model Performance Metrics")
    y_pred = model.predict(X_test)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🎯 Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    col2.metric("✅ Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2f}")
    col3.metric("📈 Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2f}")
    col4.metric("🔥 F1-Score", f"{f1_score(y_test, y_pred, average='weighted'):.2f}")
    col5.metric("📊 ROC-AUC", f"{roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'):.2f}")

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature Importance
    st.subheader("🔍 Feature Importance")
    feature_importance = pd.DataFrame({"Feature": df.columns[:-1], "Importance": model.feature_importances_})
    fig = px.bar(feature_importance.sort_values(by="Importance", ascending=False),
                 x="Importance", y="Feature", orientation='h', title="Feature Importance", color="Importance")
    st.plotly_chart(fig)

    # Product Scrap Rates
    st.subheader("♻ Product Scrap Rates")
    scrap_rate = (y_test != y_pred).mean()
    st.write(f"**Scrap Rate:** {scrap_rate:.2%}")

    # Class-wise Evaluation Metrics
    st.subheader("📋 Class-wise Evaluation Metrics")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(class_report).transpose())

    # ANOVA Plot (Permutation Importance)
    st.subheader("📊 ANOVA Plot (Permutation Importance)")
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importance = pd.DataFrame({"Feature": df.columns[:-1], "Importance": result.importances_mean})
    fig = px.bar(perm_importance.sort_values(by="Importance", ascending=False),
                 x="Importance", y="Feature", orientation='h', title="ANOVA Feature Importance", color="Importance")
    st.plotly_chart(fig)

    # Download Report Button
    st.subheader("📥 Download Model Report")
    report_df = pd.DataFrame(class_report).transpose()
    csv = report_df.to_csv(index=True)
    st.download_button(label="📩 Download CSV", data=csv, file_name="model_report.csv", mime="text/csv")

# Run the app
if __name__ == "__main__":
    main()
