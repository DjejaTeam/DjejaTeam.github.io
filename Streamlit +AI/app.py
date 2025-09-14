# ============================================================================
# POULTRY BIOGAS ANALYSIS STREAMLIT APP
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
from datetime import datetime

# ============================================================================
# LOAD MODEL CLASS (SIMPLIFIED FOR ANALYSIS APP)
# ============================================================================

class PoultryBiogasModel:
    """Poultry Biogas Model for Analysis App"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        self.model_performance = {}
        self.model_metadata = {}
    
    def create_poultry_features(self, df):
        """Create features specific to poultry biogas production"""
        # Daily waste production (0.11 kg per bird per day)
        df['Poultry_Daily_Waste_kg'] = df['Poultry'] * 0.11
        
        # Expected biogas potential (0.45 m³/kg, converted to cu-ft)
        df['Expected_Biogas_Potential'] = df['Poultry_Daily_Waste_kg'] * 0.45 * 35.315
        
        # Operation scale categories
        df['Poultry_Scale'] = pd.cut(df['Poultry'], 
                                   bins=[0, 10000, 50000, 200000, float('inf')],
                                   labels=['Small', 'Medium', 'Large', 'Industrial'],
                                   right=False)
        
        # Calculate operational years if not present
        if 'Operational Years' not in df.columns and 'Year Operational' in df.columns:
            current_year = 2024
            df['Operational Years'] = current_year - df['Year Operational']
            df['Operational Years'] = np.maximum(df['Operational Years'], 1)
        
        # Ensure Total_Animals equals Poultry
        df['Total_Animals'] = df['Poultry']
        
        return df
    
    def load_model(self, model_dir='saved_models'):
        """Load the saved model"""
        try:
            model_path = os.path.join(model_dir, 'latest_poultry_biogas_model.joblib')
            scaler_path = os.path.join(model_dir, 'latest_poultry_biogas_scaler.joblib')
            metadata_path = os.path.join(model_dir, 'latest_poultry_biogas_metadata.json')
            
            # Check if files exist
            if not all(os.path.exists(path) for path in [model_path, scaler_path, metadata_path]):
                return False, "Model files not found. Please train a model first using train_model.py"
            
            # Load model components
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.model_performance = metadata['model_performance']
            self.model_metadata = metadata['model_metadata']
            self.is_trained = True
            
            return True, "Model loaded successfully!"
            
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def predict_biogas(self, poultry_count, digester_type='Complete Mix', 
                      co_digestion='No', operational_years=5, **kwargs):
        """Make prediction"""
        if not self.is_trained:
            raise ValueError("Model not loaded!")
        
        # Create input data
        input_data = pd.DataFrame({
            'Poultry': [poultry_count],
            'Digester Type': [digester_type],
            'Co-Digestion': [co_digestion],
            'Operational Years': [operational_years],
            'Project Type': [kwargs.get('project_type', 'New')],
            'Status': [kwargs.get('status', 'Operational')],
            'Year Operational': [2024 - operational_years],
            'Biogas End Use(s)': [kwargs.get('biogas_end_use', 'Electricity')],
            'LCFS Pathway?': [kwargs.get('lcfs_pathway', 'No')],
            'Awarded USDA Funding?': [kwargs.get('usda_funding', 'No')],
            'Cattle': [0],
            'Dairy': [0],
            'Swine': [0],
            'Electricity Generated (kWh/yr)': [kwargs.get('electricity_kwh', 0)],
            'Total Emission Reductions (MTCO2e/yr)': [kwargs.get('emissions', 0)]
        })
        
        # Process input
        input_data = self.create_poultry_features(input_data)
        target_col = 'Biogas Generation Estimate (cu-ft/day)'
        if target_col in input_data.columns:
            input_data = input_data.drop(columns=[target_col])
        
        # Encode and prepare
        categorical_cols = input_data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
        else:
            input_encoded = input_data.copy()
        
        # Align features
        for col in self.feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        input_encoded = input_encoded[self.feature_names]
        input_scaled = self.scaler.transform(input_encoded)
        
        # Predict
        predicted_biogas = self.model.predict(input_scaled)[0]
        
        # Calculate metrics
        return self.calculate_metrics(poultry_count, predicted_biogas)
    
    def calculate_metrics(self, poultry_count, predicted_biogas):
        """Calculate comprehensive metrics"""
        daily_waste_kg = poultry_count * 0.11
        annual_biogas_cuft = predicted_biogas * 365
        
        # Gas composition
        methane_content = 0.60
        daily_methane_cuft = predicted_biogas * methane_content
        daily_co2_cuft = predicted_biogas * 0.35
        
        # Energy calculations
        daily_energy_btu = predicted_biogas * 600
        annual_energy_mmbtu = daily_energy_btu * 365 / 1_000_000
        
        # Economics
        energy_price_per_mmbtu = 5.0
        annual_energy_value = annual_energy_mmbtu * energy_price_per_mmbtu
        
        # Environmental
        daily_methane_kg = daily_methane_cuft * 0.0007168
        annual_co2_equivalent_avoided = daily_methane_kg * 365 * 28 / 1000
        
        # Efficiency
        theoretical_max_biogas = daily_waste_kg * 0.45 * 35.315
        system_efficiency = (predicted_biogas / theoretical_max_biogas * 100) if theoretical_max_biogas > 0 else 0
        
        return {
            'daily_biogas_cuft': round(predicted_biogas, 0),
            'annual_biogas_cuft': round(annual_biogas_cuft, 0),
            'daily_methane_cuft': round(daily_methane_cuft, 0),
            'daily_co2_cuft': round(daily_co2_cuft, 0),
            'daily_waste_kg': round(daily_waste_kg, 1),
            'annual_waste_kg': round(daily_waste_kg * 365, 0),
            'daily_energy_btu': round(daily_energy_btu, 0),
            'annual_energy_mmbtu': round(annual_energy_mmbtu, 1),
            'annual_energy_value_usd': round(annual_energy_value, 0),
            'annual_co2_equivalent_avoided_tonnes': round(annual_co2_equivalent_avoided, 1),
            'system_efficiency_percent': round(system_efficiency, 1),
            'biogas_per_bird_cuft_day': round(predicted_biogas / poultry_count, 3),
            'waste_per_bird_kg_day': round(0.11, 3),
            'theoretical_max_cuft_day': round(theoretical_max_biogas, 0)
        }

# ============================================================================
# STREAMLIT APP FUNCTIONS
# ============================================================================

def main():
    st.set_page_config(
        page_title="Poultry Biogas Analyzer", 
        layout="wide", 
        initial_sidebar_state="expanded",
        menu_items={
            'Report a bug': 'mailto:support@example.com',
            'About': "# Poultry Biogas Analyzer\nML-powered biogas analysis for poultry operations"
        }
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🐔 Poultry Biogas Analyzer</h1>
        <p style="color: white; margin: 0; opacity: 0.9;">Advanced Analysis and Prediction Tool for Poultry Farm Biogas Production</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = PoultryBiogasModel()
        success, message = st.session_state.model.load_model()
        st.session_state.model_loaded = success
        st.session_state.load_message = message
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    st.sidebar.markdown("---")
    
    # Model status
    st.sidebar.subheader("🤖 Model Status")
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model Loaded")
        if hasattr(st.session_state.model, 'model_performance'):
            perf = st.session_state.model.model_performance
            st.sidebar.info(f"**Model:** {perf.get('Model', 'Unknown')}")
            st.sidebar.info(f"**R² Score:** {perf.get('R²', 0):.3f}")
            st.sidebar.info(f"**RMSE:** {perf.get('RMSE', 0):,.0f}")
    else:
        st.sidebar.error("❌ Model Not Loaded")
        st.sidebar.warning(st.session_state.load_message)
    
    st.sidebar.markdown("---")
    
    # Navigation
    if st.session_state.model_loaded:
        page = st.sidebar.selectbox(
            "Select Analysis Mode:",
            ["🔮 Predictions", "🔄 Batch Analysis", "⚖️ Scenario Comparison", 
             "📊 Model Info", "🧮 Calculator"],
            help="Choose the analysis functionality"
        )
    else:
        st.error("⚠️ Model not available. Please train a model first using `train_model.py`")
        st.markdown("""
        ### How to get started:
        1. Run `python train_model.py` with your dataset
        2. This will create saved model files in the `saved_models/` directory
        3. Restart this app to load the trained model
        """)
        return
    
    # Main content routing
    if page == "🔮 Predictions":
        prediction_page()
    elif page == "🔄 Batch Analysis":
        batch_analysis_page()
    elif page == "⚖️ Scenario Comparison":
        comparison_page()
    elif page == "📊 Model Info":
        model_info_page()
    elif page == "🧮 Calculator":
        calculator_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <small>
    📧 **Support**: djejateam@gmail.com<br>
   </small>
    """, unsafe_allow_html=True)

def prediction_page():
    st.header("🔮 Biogas Production Prediction")
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("🐔 Operation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            poultry_count = st.number_input(
                "Number of Poultry Birds",
                min_value=100,
                max_value=1000000,
                value=25000,
                step=1000,
                help="Total number of chickens/poultry birds"
            )
            
            digester_type = st.selectbox(
                "Digester Type",
                options=['Complete Mix', 'Covered Lagoon', 'Plug Flow', 'Fixed Film'],
                help="Type of anaerobic digester system"
            )
            
            co_digestion = st.selectbox(
                "Co-Digestion",
                options=['No', 'Yes'],
                help="Adding other organic waste to digester"
            )
        
        with col2:
            operational_years = st.slider(
                "Operational Years",
                min_value=1,
                max_value=20,
                value=5,
                help="Years the system has been operational"
            )
            
            project_type = st.selectbox(
                "Project Type",
                options=['New', 'Expansion', 'Retrofit'],
                help="Type of biogas project"
            )
            
            biogas_end_use = st.selectbox(
                "Biogas End Use",
                options=['Electricity', 'Heat', 'Both', 'Fuel'],
                help="Primary use of produced biogas"
            )
        
        submitted = st.form_submit_button("🚀 Predict Biogas Production", type="primary")
    
    if submitted:
        with st.spinner("🔄 Calculating biogas production..."):
            try:
                results = st.session_state.model.predict_biogas(
                    poultry_count=poultry_count,
                    digester_type=digester_type,
                    co_digestion=co_digestion,
                    operational_years=operational_years,
                    project_type=project_type,
                    biogas_end_use=biogas_end_use
                )
                
                display_prediction_results(results, poultry_count, digester_type)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

def display_prediction_results(results, poultry_count, digester_type):
    st.success("✅ Prediction completed successfully!")
    
    # Key metrics
    st.subheader("📊 Key Production Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Daily Biogas", f"{results['daily_biogas_cuft']:,} cu-ft")
    with col2:
        st.metric("Annual Biogas", f"{results['annual_biogas_cuft']:,} cu-ft")
    with col3:
        st.metric("Daily Energy", f"{results['daily_energy_btu']:,} BTU")
    with col4:
        st.metric("System Efficiency", f"{results['system_efficiency_percent']}%")
    
    # Tabs for detailed results
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Production", "💰 Economics", "🌱 Environment", "📋 Summary"])
    
    with tab1:
        st.subheader("Gas Production Analysis")
        
        # Gas composition pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Methane (60%)', 'CO₂ (35%)', 'Other (5%)'],
            values=[results['daily_methane_cuft'], results['daily_co2_cuft'], 
                   results['daily_biogas_cuft'] - results['daily_methane_cuft'] - results['daily_co2_cuft']],
            hole=0.4,
            marker_colors=['#2E8B57', '#CD5C5C', '#4682B4']
        )])
        fig.update_layout(title="Daily Gas Composition", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Daily Methane:** {results['daily_methane_cuft']:,} cu-ft")
            st.info(f"**Per Bird Production:** {results['biogas_per_bird_cuft_day']} cu-ft/day")
            st.info(f"**Daily Waste:** {results['daily_waste_kg']} kg")
        with col2:
            st.info(f"**Daily CO₂:** {results['daily_co2_cuft']:,} cu-ft")
            st.info(f"**Annual Waste:** {results['annual_waste_kg']:,} kg")
            st.info(f"**Theoretical Max:** {results['theoretical_max_cuft_day']:,} cu-ft/day")
    
    with tab2:
        st.subheader("Economic Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Annual Energy Value", f"${results['annual_energy_value_usd']:,}")
            st.metric("Annual Energy", f"{results['annual_energy_mmbtu']} MMBtu")
            st.info("Energy price: $5.00/MMBtu")
            
        with col2:
            # 10-year projection
            years = list(range(1, 11))
            annual_values = [results['annual_energy_value_usd']] * 10
            cumulative_values = np.cumsum(annual_values)
            
            fig = px.line(
                x=years, y=cumulative_values,
                title="10-Year Cumulative Energy Value",
                labels={'x': 'Year', 'y': 'Cumulative Value ($)'}
            )
            fig.update_traces(line=dict(color='green', width=3))
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**10-Year Value:** ${results['annual_energy_value_usd']*10:,}")
    
    with tab3:
        st.subheader("Environmental Impact")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Annual CO₂ Avoided", 
                f"{results['annual_co2_equivalent_avoided_tonnes']} tonnes",
                help="CO₂ equivalent emissions avoided per year"
            )
            
            equivalent_cars = results['annual_co2_equivalent_avoided_tonnes'] * 0.22
            st.info(f"**Equivalent to removing ~{equivalent_cars:.0f} cars from road annually**")
            
        with col2:
            # Environmental impact over time
            impact_years = list(range(1, 11))
            annual_co2_avoided = [results['annual_co2_equivalent_avoided_tonnes']] * 10
            cumulative_co2 = np.cumsum(annual_co2_avoided)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=impact_years, y=cumulative_co2,
                mode='lines+markers', name='Cumulative CO₂ Avoided',
                line=dict(color='green', width=3)
            ))
            fig.update_layout(
                title="10-Year Environmental Impact",
                xaxis_title="Year", yaxis_title="Cumulative CO₂ Avoided (tonnes)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Complete Results Summary")
        
        # Operation details
        st.write("**Operation Configuration:**")
        st.write(f"- Poultry Count: {poultry_count:,} birds")
        st.write(f"- Digester Type: {digester_type}")
        st.write(f"- System Efficiency: {results['system_efficiency_percent']}%")
        
        # Results table
        results_display = {
            'Metric': [
                'Daily Biogas (cu-ft)', 'Annual Biogas (cu-ft)', 'Daily Methane (cu-ft)',
                'Daily Energy (BTU)', 'Annual Energy (MMBtu)', 'Annual Energy Value ($)',
                'CO₂ Avoided (tonnes/yr)', 'System Efficiency (%)'
            ],
            'Value': [
                f"{results['daily_biogas_cuft']:,}",
                f"{results['annual_biogas_cuft']:,}",
                f"{results['daily_methane_cuft']:,}",
                f"{results['daily_energy_btu']:,}",
                f"{results['annual_energy_mmbtu']}",
                f"${results['annual_energy_value_usd']:,}",
                f"{results['annual_co2_equivalent_avoided_tonnes']}",
                f"{results['system_efficiency_percent']}%"
            ]
        }
        
        st.dataframe(pd.DataFrame(results_display), use_container_width=True, hide_index=True)

def batch_analysis_page():
    st.header("🔄 Batch Analysis")
    st.info("Upload a CSV file with multiple operation parameters for batch predictions")
    
    # Template download
    template_data = pd.DataFrame({
        'Operation_Name': ['Farm_A', 'Farm_B', 'Farm_C'],
        'Poultry_Count': [10000, 25000, 50000],
        'Digester_Type': ['Complete Mix', 'Covered Lagoon', 'Plug Flow'],
        'Co_Digestion': ['No', 'Yes', 'No'],
        'Operational_Years': [3, 5, 8]
    })
    
    csv_template = template_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Template CSV",
        data=csv_template,
        file_name="batch_analysis_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload batch analysis file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🚀 Run Batch Analysis", type="primary"):
                with st.spinner("Processing batch analysis..."):
                    results_list = []
                    
                    progress_bar = st.progress(0)
                    total_rows = len(batch_df)
                    
                    for idx, row in batch_df.iterrows():
                        try:
                            result = st.session_state.model.predict_biogas(
                                poultry_count=row.get('Poultry_Count', 25000),
                                digester_type=row.get('Digester_Type', 'Complete Mix'),
                                co_digestion=row.get('Co_Digestion', 'No'),
                                operational_years=row.get('Operational_Years', 5)
                            )
                            
                            result['Operation_Name'] = row.get('Operation_Name', f'Operation_{idx+1}')
                            result['Input_Poultry_Count'] = row.get('Poultry_Count', 25000)
                            results_list.append(result)
                            
                        except Exception as e:
                            st.error(f"Error processing row {idx+1}: {str(e)}")
                            continue
                        
                        progress_bar.progress((idx + 1) / total_rows)
                    
                    if results_list:
                        results_df = pd.DataFrame(results_list)
                        
                        st.success(f"✅ Processed {len(results_df)} operations successfully!")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Daily Biogas", f"{results_df['daily_biogas_cuft'].sum():,.0f} cu-ft")
                        with col2:
                            st.metric("Average Daily Biogas", f"{results_df['daily_biogas_cuft'].mean():,.0f} cu-ft")
                        with col3:
                            st.metric("Total Annual Value", f"${results_df['annual_energy_value_usd'].sum():,.0f}")
                        with col4:
                            st.metric("Average Efficiency", f"{results_df['system_efficiency_percent'].mean():.1f}%")
                        
                        # Results table
                        st.subheader("📊 Detailed Results")
                        display_cols = ['Operation_Name', 'Input_Poultry_Count', 'daily_biogas_cuft', 
                                      'annual_energy_value_usd', 'system_efficiency_percent']
                        st.dataframe(results_df[display_cols], use_container_width=True)
                        
                        # Visualization
                        fig = px.bar(
                            results_df, x='Operation_Name', y='daily_biogas_cuft',
                            title="Daily Biogas Production by Operation",
                            labels={'daily_biogas_cuft': 'Daily Biogas (cu-ft)'}
                        )
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv_results,
                            file_name="batch_analysis_results.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def comparison_page():
    st.header("⚖️ Scenario Comparison")
    st.info("Compare different poultry operation scenarios side by side")
    
    scenarios = {}
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏠 Scenario A")
        scenarios['A'] = {
            'poultry_count': st.number_input("Birds (A)", value=15000, key='birds_a'),
            'digester_type': st.selectbox("Digester (A)", 
                ['Complete Mix', 'Covered Lagoon', 'Plug Flow'], key='dig_a'),
            'co_digestion': st.selectbox("Co-Digestion (A)", ['No', 'Yes'], key='co_a'),
            'operational_years': st.slider("Years (A)", 1, 20, 3, key='years_a')
        }
    
    with col2:
        st.subheader("🏭 Scenario B")
        scenarios['B'] = {
            'poultry_count': st.number_input("Birds (B)", value=25000, key='birds_b'),
            'digester_type': st.selectbox("Digester (B)", 
                ['Complete Mix', 'Covered Lagoon', 'Plug Flow'], index=1, key='dig_b'),
            'co_digestion': st.selectbox("Co-Digestion (B)", ['No', 'Yes'], index=1, key='co_b'),
            'operational_years': st.slider("Years (B)", 1, 20, 5, key='years_b')
        }
    
    with col3:
        st.subheader("🏢 Scenario C")
        scenarios['C'] = {
            'poultry_count': st.number_input("Birds (C)", value=50000, key='birds_c'),
            'digester_type': st.selectbox("Digester (C)", 
                ['Complete Mix', 'Covered Lagoon', 'Plug Flow'], index=2, key='dig_c'),
            'co_digestion': st.selectbox("Co-Digestion (C)", ['No', 'Yes'], key='co_c'),
            'operational_years': st.slider("Years (C)", 1, 20, 8, key='years_c')
        }
    
    if st.button("🔄 Compare Scenarios", type="primary"):
        with st.spinner("Calculating scenarios..."):
            comparison_results = {}
            
            for scenario_name, params in scenarios.items():
                try:
                    result = st.session_state.model.predict_biogas(**params)
                    comparison_results[scenario_name] = result
                except Exception as e:
                    st.error(f"Error in Scenario {scenario_name}: {str(e)}")
            
            if comparison_results:
                # Comparison table
                comparison_df = pd.DataFrame(comparison_results).T
                key_metrics = ['daily_biogas_cuft', 'annual_energy_value_usd', 
                             'system_efficiency_percent', 'annual_co2_equivalent_avoided_tonnes']
                
                display_df = comparison_df[key_metrics].copy()
                display_df.columns = ['Daily Biogas (cu-ft)', 'Annual Value ($)', 
                                    'Efficiency (%)', 'CO₂ Avoided (tonnes/yr)']
                
                st.subheader("📊 Scenario Comparison Results")
                st.dataframe(display_df.round(2), use_container_width=True)
                
                # Best performers
                best_biogas = max(comparison_results.items(), key=lambda x: x[1]['daily_biogas_cuft'])
                best_value = max(comparison_results.items(), key=lambda x: x[1]['annual_energy_value_usd'])
                best_efficiency = max(comparison_results.items(), key=lambda x: x[1]['system_efficiency_percent'])
                
                st.subheader("🏆 Best Performers")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.success(f"**Highest Biogas:**\nScenario {best_biogas[0]}")
                    st.info(f"{best_biogas[1]['daily_biogas_cuft']:,} cu-ft/day")
                
                with col2:
                    st.success(f"**Highest Value:**\nScenario {best_value[0]}")
                    st.info(f"${best_value[1]['annual_energy_value_usd']:,}/year")
                
                with col3:
                    st.success(f"**Highest Efficiency:**\nScenario {best_efficiency[0]}")
                    st.info(f"{best_efficiency[1]['system_efficiency_percent']}%")

def model_info_page():
    st.header("📊 Model Information")
    
    if hasattr(st.session_state.model, 'model_metadata'):
        metadata = st.session_state.model.model_metadata
        performance = st.session_state.model.model_performance
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Model Details")
            st.info(f"**Best Model:** {performance.get('Model', 'Unknown')}")
            st.info(f"**Training Date:** {metadata.get('training_date', 'Unknown')}")
            st.info(f"**Training Samples:** {metadata.get('training_samples', 'Unknown')}")
            st.info(f"**Feature Count:** {metadata.get('feature_count', 'Unknown')}")
        
        with col2:
            st.subheader("📈 Performance Metrics")
            st.metric("R² Score", f"{performance.get('R²', 0):.3f}")
            st.metric("RMSE", f"{performance.get('RMSE', 0):,.0f}")
            st.metric("MAE", f"{performance.get('MAE', 0):,.0f}")
        
        if hasattr(st.session_state.model, 'feature_names'):
            st.subheader("🔧 Model Features")
            features_df = pd.DataFrame({
                'Feature Name': st.session_state.model.feature_names
            })
            st.dataframe(features_df, use_container_width=True)

def calculator_page():
    st.header("🧮 Biogas Calculator")
    st.info("Quick calculations and theoretical estimates")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Theoretical Calculations")
        
        birds = st.number_input("Number of Birds", value=25000, step=1000)
        
        # Theoretical calculations
        daily_waste = birds * 0.11  # kg per day
        theoretical_biogas = daily_waste * 0.45 * 35.315  # cu-ft per day
        annual_biogas = theoretical_biogas * 365
        
        st.metric("Daily Waste", f"{daily_waste:,.1f} kg")
        st.metric("Theoretical Daily Biogas", f"{theoretical_biogas:,.0f} cu-ft")
        st.metric("Theoretical Annual Biogas", f"{annual_biogas:,.0f} cu-ft")
        
        # Energy content
        daily_energy_btu = theoretical_biogas * 600
        annual_energy_mmbtu = daily_energy_btu * 365 / 1_000_000
        
        st.metric("Daily Energy Content", f"{daily_energy_btu:,.0f} BTU")
        st.metric("Annual Energy Content", f"{annual_energy_mmbtu:.1f} MMBtu")
    
    with col2:
        st.subheader("💰 Economic Estimates")
        
        energy_price = st.number_input("Energy Price ($/MMBtu)", value=5.0, step=0.5)
        
        annual_value = annual_energy_mmbtu * energy_price
        
        st.metric("Annual Energy Value", f"${annual_value:,.0f}")
        st.metric("10-Year Value", f"${annual_value * 10:,.0f}")
        
        # Environmental
        methane_cuft_daily = theoretical_biogas * 0.60
        methane_kg_daily = methane_cuft_daily * 0.0007168
        co2_equivalent_annual = methane_kg_daily * 365 * 28 / 1000
        
        st.metric("Daily Methane", f"{methane_cuft_daily:,.0f} cu-ft")
        st.metric("Annual CO₂ Avoided", f"{co2_equivalent_annual:.1f} tonnes")

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()