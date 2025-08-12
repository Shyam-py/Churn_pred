import gradio as gr
import h2o
import pandas as pd

# Start H2O
h2o.init()

# Load the trained model
model = h2o.load_model(r"C:\Users\durga\Desktop\soft tools\proj-2\StackedEnsemble_BestOfFamily_1_AutoML_1_20250807_161218")

# Prediction function
def predict_churn(gender, SeniorCitizen, Partner, Dependents,
                  tenure, PhoneService, MultipleLines, InternetService,
                  OnlineSecurity, OnlineBackup, DeviceProtection,
                  TechSupport, StreamingTV, StreamingMovies, Contract,
                  PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    try:
        # Build DataFrame
        data = pd.DataFrame([[
            gender, SeniorCitizen, Partner, Dependents,
            tenure, PhoneService, MultipleLines, InternetService,
            OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies, Contract,
            PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
        ]], columns=[
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ])

        # Ensure correct types
        numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Convert to H2O Frame
        h2o_data = h2o.H2OFrame(data)

        # Convert strings to categorical
        for col in data.select_dtypes(include='object').columns:
            h2o_data[col] = h2o_data[col].asfactor()

        # Predict
        prediction = model.predict(h2o_data).as_data_frame()
        result = prediction['predict'][0]
        prob = round(prediction['p1'][0] * 100, 2)
        return f"Prediction: {'Churn' if result == '1' else 'No Churn'} ({prob}% confidence)"
    
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Gradio Inputs
gender = gr.Dropdown(["Male", "Female"], label="Gender")
SeniorCitizen = gr.Radio([0, 1], label="Senior Citizen (0 = No, 1 = Yes)")
Partner = gr.Dropdown(["Yes", "No"], label="Partner")
Dependents = gr.Dropdown(["Yes", "No"], label="Dependents")
tenure = gr.Slider(0, 72, step=1, label="Tenure (Months)")
PhoneService = gr.Dropdown(["Yes", "No"], label="Phone Service")
MultipleLines = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines")
InternetService = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service")
OnlineSecurity = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security")
OnlineBackup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup")
DeviceProtection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection")
TechSupport = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support")
StreamingTV = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV")
StreamingMovies = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies")
Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract")
PaperlessBilling = gr.Dropdown(["Yes", "No"], label="Paperless Billing")
PaymentMethod = gr.Dropdown([
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
], label="Payment Method")
MonthlyCharges = gr.Number(label="Monthly Charges")
TotalCharges = gr.Number(label="Total Charges")

# Launch Interface
gr.Interface(
    fn=predict_churn,
    inputs=[gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges],
    outputs="text",
    title="Customer Churn Predictor",
    description="AI-powered prediction using H2O AutoML"
).launch()
