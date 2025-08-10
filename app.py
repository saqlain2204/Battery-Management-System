import streamlit as st
import pandas as pd
import joblib
import os
import json
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Configuration file path
CONFIG_PATH = 'config/model_config.json'

# Load configuration
@st.cache_data
def load_config():
    """Load model and dataset configuration from JSON"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        st.error(f"Configuration file not found: {CONFIG_PATH}")
        return None
    except json.JSONDecodeError:
        st.error(f"Invalid JSON format in: {CONFIG_PATH}")
        return None

# Find the latest model file in the models directory
def get_latest_model(models_dir='models'):
    """Get the production model from config"""
    config = load_config()
    if config and 'production_model' in config:
        production_model_id = config['production_model']['model_id']
        # Find the model with this ID
        for model in config['models']:
            if model['model_id'] == production_model_id:
                model_file = model['model_file']
                model_path = os.path.join(models_dir, model_file)
                if os.path.exists(model_path):
                    return model_path
    return None

# Load model results from JSON config
def load_model_results():
    """Load the latest model results from JSON config"""
    try:
        config = load_config()
        if config and 'production_model' in config:
            production_model_id = config['production_model']['model_id']
            # Find the production model
            for model in config['models']:
                if model['model_id'] == production_model_id:
                    return model
        return None
    except Exception as e:
        st.error(f"Error loading model results: {e}")
        return None

# Get all models from config
def get_all_models():
    """Get all models from config"""
    config = load_config()
    if config and 'models' in config:
        return config['models']
    return []

# Get dataset info from config
def get_dataset_info():
    """Get dataset information from config"""
    config = load_config()
    if config and 'dataset_info' in config:
        return config['dataset_info']
    return None

# Load the specific LinearRegression model
def load_linear_regression_model():
    """Load the LinearRegression model specifically"""
    model_filename = "SOC_LinearRegression_v1.0_test_result_trial_end_v1.0_20250804_004225.joblib"
    model_path = os.path.join('models', model_filename)
    
    if os.path.exists(model_path):
        try:
            loaded_model = joblib.load(model_path)
            st.sidebar.success(f"✅ Model loaded: LinearRegression")
            return loaded_model
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")
            return None
    else:
        st.sidebar.error(f"❌ Model file not found: {model_filename}")
        return None

# Load the LinearRegression model for predictions
model = load_linear_regression_model()

# Set page config for better UI
st.set_page_config(
    page_title="SOC Prediction Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown('''
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stCard {
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        padding: 2rem;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
''', unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🔋 SOC Prediction Dashboard</h1>', unsafe_allow_html=True)

# Authentication check
def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def login():
    """Handle user login"""
    st.markdown("## 🔐 Login Required")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            # Check credentials against secrets
            if (username == st.secrets["auth"]["username"] and 
                password == st.secrets["auth"]["password"]):
                st.session_state['authenticated'] = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

# Check if user is authenticated
if not check_authentication():
    login()
    st.stop()

# If authenticated, show the main app

# Sidebar with enhanced styling
st.sidebar.markdown("### 📊 Navigation")
menu = st.sidebar.radio('Choose Section:', [
    '📈 Model Evaluation',
    '🔮 Predict SOC',
    '📊 Dataset Visualization'
], index=0)

# Add sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("This dashboard provides SOC prediction capabilities using machine learning models trained on battery data.")

# Add logout button
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state['authenticated'] = False
    st.rerun()

if menu == '📈 Model Evaluation':
    st.markdown("## 📈 Model Performance Metrics")
    
    # Load results from CSV
    latest_results = load_model_results()
    
    if latest_results is not None:
        # Display metrics from JSON config in enhanced cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('''
                <div class="metric-card">
                    <h3>🎯 RMSE</h3>
                    <h2>{:.4f}</h2>
                    <p>Root Mean Squared Error</p>
                </div>
            '''.format(latest_results['performance']['rmse']), unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
                <div class="metric-card">
                    <h3>📊 R² Score</h3>
                    <h2>{:.4f}</h2>
                    <p>Coefficient of Determination</p>
                </div>
            '''.format(latest_results['performance']['r2_score']), unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
                <div class="metric-card">
                    <h3>� MAE</h3>
                    <h2>{:.4f}</h2>
                    <p>Mean Absolute Error</p>
                </div>
            '''.format(latest_results['performance']['mae']), unsafe_allow_html=True)
        
        # Add detailed model info
        st.markdown("### 🤖 Model Information")
        model_info_col1, model_info_col2 = st.columns(2)
        with model_info_col1:
            st.info(f"**Model Type:** {latest_results['model_name']}")
            st.info(f"**Dataset:** {latest_results['dataset_version']}")
            st.info(f"**Features Used:** {', '.join(latest_results['features_used'])}")
        with model_info_col2:
            st.info(f"**Training Size:** {latest_results['train_size']:,} samples")
            st.info(f"**Test Size:** {latest_results['test_size']:,} samples")
            st.info(f"**Model Trained:** {latest_results['timestamp']}")
        
        # Show all model results if multiple exist
        all_models = get_all_models()
        if len(all_models) > 1:
            st.markdown("### 📊 All Model Results")
            
            # Convert to DataFrame for display
            models_data = []
            for model in all_models:
                models_data.append({
                    'model_name': model['model_name'],
                    'dataset_name': model['dataset_version'],
                    'r2_score': model['performance']['r2_score'],
                    'rmse': model['performance']['rmse'],
                    'mae': model['performance']['mae'],
                    'timestamp': model['timestamp'],
                    'status': model['status']
                })
            
            df_models = pd.DataFrame(models_data)
            st.dataframe(df_models.round(4), use_container_width=True)
            
            # Plot comparison if multiple models
            st.markdown("### 📈 Model Comparison")
            metrics_to_plot = ['r2_score', 'rmse', 'mae']
            selected_metric = st.selectbox("Select metric to compare:", metrics_to_plot)
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            
            model_names = [model['model_name'] for model in all_models]
            metric_values = [model['performance'][selected_metric] for model in all_models]
            
            bars = ax.bar(range(len(model_names)), metric_values)
            ax.set_xlabel('Model')
            ax.set_ylabel(selected_metric.upper())
            ax.set_title(f'Model Comparison - {selected_metric.upper()}')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([f"{name}\n{all_models[i]['timestamp']}" for i, name in enumerate(model_names)], 
                             rotation=45, ha='right')
            
            # Color bars based on performance
            if selected_metric == 'r2_score':
                colors = ['green' if x == max(metric_values) else 'lightblue' for x in metric_values]
            else:
                colors = ['green' if x == min(metric_values) else 'lightcoral' for x in metric_values]
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning('⚠️ No model found in models directory.')

elif menu == '🔮 Predict SOC':
    st.markdown("## 🔮 SOC Prediction")
    
    # Display temperature range info from model config
    latest_results = load_model_results()
    if latest_results and 'dataset_info' in latest_results:
        dataset_info = latest_results['dataset_info']
        if 'temperature_range' in dataset_info:
            min_temp = dataset_info['temperature_range']['min']
            max_temp = dataset_info['temperature_range']['max']
            st.markdown(f'''
                <div style="background: transparent; border: 2px solid #1f77b4; border-radius: 15px; padding: 1.5rem; margin: 1rem 0;">
                    <h4 style="color: #1f77b4; margin: 0 0 0.5rem 0;">🌡️ Temperature Range in Dataset</h4>
                    <p style="color: black; font-weight: bold; margin: 0;"><strong>{min_temp:.2f}°C</strong> to <strong>{max_temp:.2f}°C</strong></p>
                </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("### 📝 Enter Input Values")
    
    # Input form with better layout
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            time = st.number_input('⏰ Time', min_value=0.0, value=0.0, help="Time value for prediction")
            voltage = st.number_input('⚡ Voltage (V)', min_value=0.0, value=3.7, help="Battery voltage in volts")
        with col2:
            current = st.number_input('🔌 Current (A)', value=0.0, help="Battery current in amperes")
            max_temperature = st.number_input('🌡️ Max Temperature (°C)', value=25.0, help="Maximum temperature in Celsius")
        
        submitted = st.form_submit_button("🚀 Predict SOC", use_container_width=True)
        
        if submitted:
            if model is not None:
                input_df = pd.DataFrame({
                    'time': [time],
                    'voltage': [voltage],
                    'current': [current],
                    'max_temperature': [max_temperature]
                })
                soc_pred = model.predict(input_df)[0]
                
                # Display prediction with dynamic styling based on SOC level
                st.markdown("### 🎯 Prediction Result")
                
                # Determine color based on SOC level
                if soc_pred >= 80:
                    bg_color = "linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%)"  # Green
                    status_emoji = "🟢"
                    status_text = "Excellent"
                elif soc_pred >= 60:
                    bg_color = "linear-gradient(135deg, #8BC34A 0%, #CDDC39 100%)"  # Light Green
                    status_emoji = "🟡"
                    status_text = "Good"
                elif soc_pred >= 40:
                    bg_color = "linear-gradient(135deg, #FF9800 0%, #FFC107 100%)"  # Orange
                    status_emoji = "🟠"
                    status_text = "Moderate"
                elif soc_pred >= 20:
                    bg_color = "linear-gradient(135deg, #FF5722 0%, #FF9800 100%)"  # Red-Orange
                    status_emoji = "🔴"
                    status_text = "Low"
                else:
                    bg_color = "linear-gradient(135deg, #F44336 0%, #E91E63 100%)"  # Red
                    status_emoji = "❌"
                    status_text = "Critical"
                
                st.markdown(f'''
                    <div style="background: {bg_color}; color: white; border-radius: 15px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2); text-align: center;">
                        <h2>🔋 Predicted SOC</h2>
                        <h1 style="font-size: 4rem; margin: 1rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{soc_pred:.2f}%</h1>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">{status_emoji} Battery Status: <strong>{status_text}</strong></p>
                        <p>State of Charge Prediction</p>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Add interpretation with matching colors
                if soc_pred >= 80:
                    st.success("✅ Battery is well charged!")
                elif soc_pred >= 60:
                    st.success("✅ Battery is in good condition!")
                elif soc_pred >= 40:
                    st.info("ℹ️ Battery is moderately charged.")
                elif soc_pred >= 20:
                    st.warning("⚠️ Battery is getting low.")
                else:
                    st.error("❌ Battery needs charging soon!")
            else:
                st.error('❌ No model available for prediction.')

elif menu == '📊 Dataset Visualization':
    st.markdown("## 📊 Dataset Analysis & Visualization")
    
    # Load dataset info from model config
    latest_results = load_model_results()
    if latest_results and 'dataset_info' in latest_results:
        dataset_info = latest_results['dataset_info']
        
        # Dataset overview from JSON config
        st.markdown("### 📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_records = dataset_info.get('total_records', 'N/A')
            st.metric("📊 Total Records", f"{total_records:,}" if isinstance(total_records, int) else str(total_records))
        with col2:
            features_count = len(latest_results.get('features_used', []))
            st.metric("📝 Features", f"{features_count}")
        with col3:
            if 'temperature_range' in dataset_info:
                temp_range = dataset_info['temperature_range']
                st.metric("🌡️ Temperature Range", f"{temp_range['min']:.1f}°C - {temp_range['max']:.1f}°C")
            else:
                st.metric("🌡️ Temperature Range", "N/A")
        with col4:
            if 'voltage_stats' in dataset_info:
                avg_voltage = dataset_info['voltage_stats'].get('mean', 0)
                st.metric("⚡ Avg Voltage", f"{avg_voltage:.2f}V")
            else:
                st.metric("⚡ Avg Voltage", "N/A")
        
        # Dataset Statistics
        st.markdown("### 📊 Dataset Statistics")
        
        # Create expandable sections for different statistics
        with st.expander("📈 Feature Statistics", expanded=True):
            if 'feature_stats' in dataset_info:
                stats_data = []
                for feature, stats in dataset_info['feature_stats'].items():
                    stats_data.append({
                        'Feature': feature,
                        'Min': f"{stats.get('min', 'N/A'):.4f}" if isinstance(stats.get('min'), (int, float)) else 'N/A',
                        'Max': f"{stats.get('max', 'N/A'):.4f}" if isinstance(stats.get('max'), (int, float)) else 'N/A',
                        'Mean': f"{stats.get('mean', 'N/A'):.4f}" if isinstance(stats.get('mean'), (int, float)) else 'N/A',
                        'Std': f"{stats.get('std', 'N/A'):.4f}" if isinstance(stats.get('std'), (int, float)) else 'N/A'
                    })
                
                if stats_data:
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(df_stats, use_container_width=True)
            else:
                st.info("Feature statistics not available in current model configuration.")
        
        with st.expander("🎯 Target Variable (SOC) Analysis", expanded=False):
            if 'target_stats' in dataset_info:
                target_stats = dataset_info['target_stats']
                soc_col1, soc_col2, soc_col3 = st.columns(3)
                with soc_col1:
                    st.metric("🎯 Min SOC", f"{target_stats.get('min', 'N/A'):.2f}%" if isinstance(target_stats.get('min'), (int, float)) else 'N/A')
                with soc_col2:
                    st.metric("🎯 Max SOC", f"{target_stats.get('max', 'N/A'):.2f}%" if isinstance(target_stats.get('max'), (int, float)) else 'N/A')
                with soc_col3:
                    st.metric("🎯 Avg SOC", f"{target_stats.get('mean', 'N/A'):.2f}%" if isinstance(target_stats.get('mean'), (int, float)) else 'N/A')
            else:
                st.info("Target variable statistics not available in current model configuration.")
        
        with st.expander("🔗 Data Quality Information", expanded=False):
            data_quality = dataset_info.get('data_quality', {})
            
            qual_col1, qual_col2 = st.columns(2)
            with qual_col1:
                st.info(f"**Missing Values:** {data_quality.get('missing_values', 'N/A')}")
                st.info(f"**Duplicates:** {data_quality.get('duplicates', 'N/A')}")
            with qual_col2:
                st.info(f"**Data Types:** {data_quality.get('data_types_count', 'N/A')}")
                st.info(f"**Memory Usage:** {data_quality.get('memory_usage', 'N/A')}")
        
        # Model Performance Visualization
        st.markdown("### 🎯 Model Performance")
        
        if 'performance' in latest_results:
            perf = latest_results['performance']
            
            # Create performance metrics visualization
            import matplotlib.pyplot as plt
            
            metrics = ['R² Score', 'RMSE', 'MAE']
            values = [perf.get('r2_score', 0), perf.get('rmse', 0), perf.get('mae', 0)]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar chart for metrics
            colors = ['green' if metrics[i] == 'R² Score' else 'orange' for i in range(len(metrics))]
            bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
            ax1.set_title('Model Performance Metrics')
            ax1.set_ylabel('Metric Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.4f}', ha='center', va='bottom')
            
            # Pie chart for error distribution (if available)
            if 'error_distribution' in perf:
                error_dist = perf['error_distribution']
                ax2.pie(error_dist.values(), labels=error_dist.keys(), autopct='%1.1f%%')
                ax2.set_title('Error Distribution')
            else:
                # Simple performance indicator
                performance_score = perf.get('r2_score', 0)
                colors_pie = ['green' if performance_score > 0.8 else 'orange' if performance_score > 0.6 else 'red', 'lightgray']
                sizes = [performance_score * 100, (1 - performance_score) * 100]
                ax2.pie(sizes, labels=['Model Accuracy', 'Room for Improvement'], 
                       colors=colors_pie, autopct='%1.1f%%', startangle=90)
                ax2.set_title(f'Model Accuracy Score (R²)')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # Training Information
        st.markdown("### 🏃‍♂️ Training Information")
        train_col1, train_col2, train_col3 = st.columns(3)
        with train_col1:
            st.info(f"**Model Type:** {latest_results.get('model_name', 'N/A')}")
        with train_col2:
            st.info(f"**Training Samples:** {latest_results.get('train_size', 'N/A'):,}" if isinstance(latest_results.get('train_size'), int) else f"**Training Samples:** {latest_results.get('train_size', 'N/A')}")
        with train_col3:
            st.info(f"**Test Samples:** {latest_results.get('test_size', 'N/A'):,}" if isinstance(latest_results.get('test_size'), int) else f"**Test Samples:** {latest_results.get('test_size', 'N/A')}")
    
    else:
        st.warning('⚠️ Under development. please come back later.')
