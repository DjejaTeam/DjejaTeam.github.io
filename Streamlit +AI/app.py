# ============================================================================
# POULTRY BIOGAS ANALYSIS STREAMLIT APP (Simplified + Year Feature)
# ============================================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import joblib, json, os, warnings
warnings.filterwarnings("ignore")

# ============================================================================
# MODEL CLASS
# ============================================================================
class PoultryBiogasModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
    
    def load_model(self, model_dir='saved_models'):
        try:
            model_path = os.path.join(model_dir, 'latest_poultry_biogas_model.joblib')
            scaler_path = os.path.join(model_dir, 'latest_poultry_biogas_scaler.joblib')
            metadata_path = os.path.join(model_dir, 'latest_poultry_biogas_metadata.json')
            if not all(os.path.exists(p) for p in [model_path, scaler_path, metadata_path]):
                return False, "Model files not found"
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.feature_names = metadata['feature_names']
            self.is_trained = True
            return True, "Model loaded!"
        except Exception as e:
            return False, str(e)

    def predict_biogas(self, poultry_count, digester_type='Complete Mix', co_digestion='No', operational_years=5, year_operational=2024):
        if not self.is_trained:
            raise ValueError("Model not loaded")
        input_data = pd.DataFrame({
            'Poultry': [poultry_count],
            'Digester Type': [digester_type],
            'Co-Digestion': [co_digestion],
            'Operational Years': [operational_years],
            'Year Operational': [year_operational],
            'Project Type': ['New'],
            'Status': ['Operational'],
            'Biogas End Use(s)': ['Electricity'],
            'LCFS Pathway?': ['No'],
            'Awarded USDA Funding?': ['No'],
            'Cattle': [0],
            'Dairy': [0],
            'Swine': [0]
        })
        for col in self.feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        input_encoded = input_data[self.feature_names]
        input_scaled = self.scaler.transform(input_encoded)
        predicted_cuft = self.model.predict(input_scaled)[0]
        return self.calculate_metrics(poultry_count, predicted_cuft, year_operational)

    def calculate_metrics(self, poultry_count, predicted_cuft, year_operational):
        daily_m3 = predicted_cuft * 0.0283168
        annual_m3 = daily_m3 * 365
        daily_kwh = (predicted_cuft * 600) / 3412
        annual_kwh = daily_kwh * 365
        return {
            "year_operational": year_operational,
            "daily_biogas_m3": round(daily_m3, 1),
            "annual_biogas_m3": round(annual_m3, 0),
            "daily_energy_kwh": round(daily_kwh, 1),
            "annual_energy_kwh": round(annual_kwh, 0),
            "per_bird_m3_day": round(daily_m3 / poultry_count, 4)
        }

# ============================================================================
# STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(page_title="Poultry Biogas Analyzer", layout="wide")
    st.title("🐔 Poultry Biogas Analyzer")

    if 'model' not in st.session_state:
        st.session_state.model = PoultryBiogasModel()
        ok, msg = st.session_state.model.load_model()
        st.session_state.model_loaded = ok
        st.session_state.load_message = msg

    if not st.session_state.model_loaded:
        st.error("⚠️ Model not loaded")
        st.info(st.session_state.load_message)
        return

    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox("Select Page", ["🔮 Predictions", "🔄 Batch Analysis", "⚖️ Scenario Comparison"])

    if page == "🔮 Predictions":
        prediction_page()
    elif page == "🔄 Batch Analysis":
        batch_analysis_page()
    elif page == "⚖️ Scenario Comparison":
        comparison_page()

def prediction_page():
    st.header("🔮 Biogas Production Prediction")
    poultry_count = st.number_input("Number of Poultry Birds", min_value=100, value=25000, step=1000)
    digester_type = st.selectbox("Digester Type", ['Complete Mix','Covered Lagoon','Plug Flow'])
    co_digestion = st.selectbox("Co-Digestion", ['No','Yes'])
    operational_years = st.slider("Operational Years",1,20,5)
    year_operational = st.number_input("Year Operational", min_value=2000, max_value=2100, value=2024)

    if st.button("🚀 Predict Biogas Production"):
        results = st.session_state.model.predict_biogas(
            poultry_count, digester_type, co_digestion, operational_years, year_operational
        )
        st.success("✅ Prediction completed")
        col1, col2, col3 = st.columns(3)
        col1.metric("Daily Biogas", f"{results['daily_biogas_m3']} m³")
        col2.metric("Annual Biogas", f"{results['annual_biogas_m3']:,} m³")
        col3.metric("Daily Energy", f"{results['daily_energy_kwh']} kWh")
        st.info(f"📅 Year Operational: {results['year_operational']}")

def batch_analysis_page():
    st.header("🔄 Batch Analysis")
    st.info("Upload CSV with columns: Operation_Name,Poultry_Count,Digester_Type,Co_Digestion,Operational_Years,Year_Operational")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        results = []
        for _,row in df.iterrows():
            res = st.session_state.model.predict_biogas(
                row['Poultry_Count'], row['Digester_Type'], row['Co_Digestion'],
                row['Operational_Years'], row['Year_Operational']
            )
            res['Operation_Name'] = row['Operation_Name']
            results.append(res)
        res_df = pd.DataFrame(results)
        st.dataframe(res_df)
        fig = px.bar(res_df, x="Operation_Name", y="daily_biogas_m3", title="Daily Biogas (m³)")
        st.plotly_chart(fig, use_container_width=True)

def comparison_page():
    st.header("⚖️ Scenario Comparison")
    scenarios = {
        "A": {"birds": st.number_input("Birds (A)", value=10000, key="a"),
              "year": st.number_input("Year (A)", value=2024, key="ya")},
        "B": {"birds": st.number_input("Birds (B)", value=25000, key="b"),
              "year": st.number_input("Year (B)", value=2024, key="yb")},
        "C": {"birds": st.number_input("Birds (C)", value=50000, key="c"),
              "year": st.number_input("Year (C)", value=2024, key="yc")}
    }
    if st.button("Compare"):
        results = {k: st.session_state.model.predict_biogas(v["birds"], year_operational=v["year"]) for k,v in scenarios.items()}
        df = pd.DataFrame(results).T
        st.dataframe(df)

if __name__ == "__main__":
    main()
