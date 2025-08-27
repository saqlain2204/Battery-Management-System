import streamlit as st
import pandas as pd
import joblib
import os
import json
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Configuration file path
CONFIG_PATH = 'config/model_config.json'

def compute_dataset_info(df, target_col='SOC'):
    """Compute dataset info and feature statistics from a DataFrame."""
    info = {}
    info['total_records'] = len(df)
    info['features'] = [col for col in df.columns if col != target_col]
    info['target'] = target_col

    # Temperature range
    if 'max_temperature' in df.columns:
        info['temperature_range'] = {
            'min': float(df['max_temperature'].min()),
            'max': float(df['max_temperature'].max()),
            'unit': '°C'
        }

    # Voltage stats
    if 'voltage' in df.columns:
        info['voltage_stats'] = {
            'mean': float(df['voltage'].mean()),
            'min': float(df['voltage'].min()),
            'max': float(df['voltage'].max()),
            'unit': 'V'
        }

    # Current stats
    if 'current' in df.columns:
        info['current_stats'] = {
            'mean': float(df['current'].mean()),
            'min': float(df['current'].min()),
            'max': float(df['current'].max()),
            'unit': 'A'
        }

    # Feature statistics
    feature_stats = {}
    for col in info['features']:
        feature_stats[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'unit': 'N/A'
        }
    info['feature_stats'] = feature_stats

    # Target statistics
    if target_col in df.columns:
        info['target_stats'] = {
            'min': float(df[target_col].min()),
            'max': float(df[target_col].max()),
            'mean': float(df[target_col].mean()),
            'std': float(df[target_col].std()),
            'unit': '%'
        }

    # Data quality
    info['data_quality'] = {
        'missing_values': int(df.isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum()),
        'data_types_count': f"{len(df.select_dtypes(include=np.number).columns)} numeric features",
        'memory_usage': f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
    }

    return info

def update_config_dataset_info(config_path, dataset_path, target_col='SOC'):
    """Update the dataset_info section in the config file based on the actual dataset."""
    try:
        df = pd.read_csv(dataset_path)
        dataset_info = compute_dataset_info(df, target_col=target_col)
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['dataset_info'] = dataset_info
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Updated dataset_info in config file.")
    except Exception as e:
        print(f"❌ Could not update dataset_info: {e}")

# ==================== UTILITY FUNCTIONS ====================

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

def get_production_model_info(target='SOC'):
    """Get production model info for SOC or SoH"""
    config = load_config()
    if config and 'production_model' in config:
        production_model_id = config['production_model'][f'{target.lower()}_model_id']
        # Find the model with this ID
        for model in config['models']:
            if model['model_id'] == production_model_id:
                return model
    return None

def load_model_from_config(target='SOC'):
    """Load the production model for SOC or SoH"""
    model_info = get_production_model_info(target)
    if model_info:
        model_file = model_info['model_file']
        model_path = os.path.join('models', model_file)
        if os.path.exists(model_path):
            try:
                loaded_model = joblib.load(model_path)
                st.sidebar.success(f"✅ {target} Model loaded: {model_info['model_name']}")
                return loaded_model, model_info
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {target} model: {e}")
                return None, None
        else:
            st.sidebar.error(f"❌ {target} Model file not found: {model_file}")
            return None, None
    return None, None

def get_all_models_by_target(target='SOC'):
    """Get all models for a specific target (SOC or SoH)"""
    config = load_config()
    if config and 'models' in config:
        return [model for model in config['models'] if model['target'] == target]
    return []

def get_dataset_info():
    """Get dataset information from config"""
    config = load_config()
    if config and 'dataset_info' in config:
        return config['dataset_info']
    return None

# ==================== UI COMPONENTS ====================

def setup_page_config():
    """Set up Streamlit page configuration"""
    st.set_page_config(
        page_title="Battery Management Dashboard",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    """Apply custom CSS styling"""
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
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .prediction-card {
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            text-align: center;
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

def create_sidebar():
    """Create and configure sidebar"""
    st.sidebar.markdown("### 📊 Navigation")
    menu = st.sidebar.radio('Choose Section:', [
        '📈 Model Evaluation',
        '🔮 Predict SOC',
        '🏥 Predict SoH',
        '📊 Dataset Visualization'
    ], index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("This dashboard provides SOC and SoH prediction capabilities using machine learning models trained on battery data.")
    
    # Add logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state['authenticated'] = False
        st.rerun()
    
    return menu

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

# ==================== VISUALIZATION FUNCTIONS ====================

def display_metrics_cards(model_info, target='SOC'):
    """Display performance metrics in cards"""
    if not model_info or 'performance' not in model_info:
        st.warning(f'⚠️ No {target} model performance data available.')
        return
    
    perf = model_info['performance']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
            <div class="metric-card">
                <h3>🎯 RMSE</h3>
                <h2>{perf.get('rmse', 0):.4f}</h2>
                <p>Root Mean Squared Error</p>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
            <div class="metric-card">
                <h3>📊 R² Score</h3>
                <h2>{perf.get('r2_score', 0):.4f}</h2>
                <p>Coefficient of Determination</p>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
            <div class="metric-card">
                <h3>📏 MAE</h3>
                <h2>{perf.get('mae', 0):.4f}</h2>
                <p>Mean Absolute Error</p>
            </div>
        ''', unsafe_allow_html=True)

def display_model_info(model_info, target='SOC'):
    """Display detailed model information"""
    if not model_info:
        return
    
    st.markdown(f"### 🤖 {target} Model Information")
    model_info_col1, model_info_col2 = st.columns(2)
    with model_info_col1:
        st.info(f"**Model Type:** {model_info.get('model_name', 'N/A')}")
        st.info(f"**Dataset:** {model_info.get('dataset_version', 'N/A')}")
        st.info(f"**Features Used:** {', '.join(model_info.get('features_used', []))}")
    with model_info_col2:
        st.info(f"**Training Size:** {model_info.get('train_size', 'N/A'):,}" if isinstance(model_info.get('train_size'), int) else f"**Training Size:** {model_info.get('train_size', 'N/A')}")
        st.info(f"**Test Size:** {model_info.get('test_size', 'N/A'):,}" if isinstance(model_info.get('test_size'), int) else f"**Test Size:** {model_info.get('test_size', 'N/A')}")
        st.info(f"**Model Trained:** {model_info.get('timestamp', 'N/A')}")

def display_models_comparison(models, target='SOC'):
    """Display comparison of all models"""
    if len(models) <= 1:
        return
    
    st.markdown(f"### 📊 All {target} Model Results")
    
    # Convert to DataFrame for display
    models_data = []
    for model in models:
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
    
    # Plot comparison
    st.markdown(f"### 📈 {target} Model Comparison")
    metrics_to_plot = ['r2_score', 'rmse', 'mae']
    selected_metric = st.selectbox(f"Select metric to compare for {target}:", metrics_to_plot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = [model['model_name'] for model in models]
    metric_values = [model['performance'][selected_metric] for model in models]
    
    bars = ax.bar(range(len(model_names)), metric_values)
    ax.set_xlabel('Model')
    ax.set_ylabel(selected_metric.upper())
    ax.set_title(f'{target} Model Comparison - {selected_metric.upper()}')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([f"{name}\n{models[i]['timestamp']}" for i, name in enumerate(model_names)], 
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
    plt.close(fig)

def get_prediction_styling(prediction_value, target='SOC'):
    """Get styling based on prediction value"""
    if target == 'SOC':
        if prediction_value >= 80:
            return "linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%)", "🟢", "Excellent"
        elif prediction_value >= 60:
            return "linear-gradient(135deg, #8BC34A 0%, #CDDC39 100%)", "🟡", "Good"
        elif prediction_value >= 40:
            return "linear-gradient(135deg, #FF9800 0%, #FFC107 100%)", "🟠", "Moderate"
        elif prediction_value >= 20:
            return "linear-gradient(135deg, #FF5722 0%, #FF9800 100%)", "🔴", "Low"
        else:
            return "linear-gradient(135deg, #F44336 0%, #E91E63 100%)", "❌", "Critical"
    else:  # SoH
        if prediction_value >= 80:
            return "linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%)", "🟢", "Healthy"
        elif prediction_value >= 60:
            return "linear-gradient(135deg, #8BC34A 0%, #CDDC39 100%)", "🟡", "Good"
        elif prediction_value >= 40:
            return "linear-gradient(135deg, #FF9800 0%, #FFC107 100%)", "🟠", "Degraded"
        elif prediction_value >= 20:
            return "linear-gradient(135deg, #FF5722 0%, #FF9800 100%)", "🔴", "Poor"
        else:
            return "linear-gradient(135deg, #F44336 0%, #E91E63 100%)", "❌", "Critical"

def display_prediction_result(prediction_value, target='SOC'):
    """Display prediction result with styling"""
    bg_color, status_emoji, status_text = get_prediction_styling(prediction_value, target)
    
    unit = "%" if target in ['SOC', 'SoH'] else ""
    icon = "🔋" if target == 'SOC' else "🏥"
    title = "Predicted SOC" if target == 'SOC' else "Predicted SoH"
    subtitle = "State of Charge Prediction" if target == 'SOC' else "State of Health Prediction"
    
    st.markdown(f'''
        <div class="prediction-card" style="background: {bg_color}; color: white;">
            <h2>{icon} {title}</h2>
            <h1 style="font-size: 4rem; margin: 1rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{prediction_value:.2f}{unit}</h1>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">{status_emoji} Status: <strong>{status_text}</strong></p>
            <p>{subtitle}</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # Add interpretation
    if target == 'SOC':
        if prediction_value >= 80:
            st.success("✅ Battery is well charged!")
        elif prediction_value >= 60:
            st.success("✅ Battery is in good condition!")
        elif prediction_value >= 40:
            st.info("ℹ️ Battery is moderately charged.")
        elif prediction_value >= 20:
            st.warning("⚠️ Battery is getting low.")
        else:
            st.error("❌ Battery needs charging soon!")
    else:  # SoH
        if prediction_value >= 80:
            st.success("✅ Battery health is excellent!")
        elif prediction_value >= 60:
            st.success("✅ Battery health is good!")
        elif prediction_value >= 40:
            st.info("ℹ️ Battery shows signs of degradation.")
        elif prediction_value >= 20:
            st.warning("⚠️ Battery health is poor.")
        else:
            st.error("❌ Battery replacement recommended!")

def create_prediction_form(target='SOC'):
    """Create input form for predictions"""
    st.markdown("### 📝 Enter Input Values")
    
    with st.form(f"prediction_form_{target.lower()}"):
        col1, col2 = st.columns(2)
        with col1:
            time = st.number_input('⏰ Time', min_value=0.0, value=0.0, help="Time value for prediction", key=f"time_{target}")
            voltage = st.number_input('⚡ Voltage (V)', min_value=0.0, value=3.7, help="Battery voltage in volts", key=f"voltage_{target}")
        with col2:
            current = st.number_input('🔌 Current (A)', value=0.0, help="Battery current in amperes", key=f"current_{target}")
            max_temperature = st.number_input('🌡️ Max Temperature (°C)', value=25.0, help="Maximum temperature in Celsius", key=f"temp_{target}")
        
        button_text = f"🚀 Predict {target}"
        submitted = st.form_submit_button(button_text, use_container_width=True)
        
        if submitted:
            return {
                'time': time,
                'voltage': voltage,
                'current': current,
                'max_temperature': max_temperature
            }
    return None

# ==================== PAGE FUNCTIONS ====================

def show_model_evaluation():
    """Display model evaluation page"""
    st.markdown("## 📈 Model Performance Metrics")
    
    # Create tabs for SOC and SoH
    tab1, tab2 = st.tabs(["🔋 SOC Models", "🏥 SoH Models"])
    
    with tab1:
        st.markdown("### SOC Model Performance")
        soc_model_info = get_production_model_info('SOC')
        display_metrics_cards(soc_model_info, 'SOC')
        display_model_info(soc_model_info, 'SOC')
        
        all_soc_models = get_all_models_by_target('SOC')
        display_models_comparison(all_soc_models, 'SOC')
    
    with tab2:
        st.markdown("### SoH Model Performance")
        soh_model_info = get_production_model_info('SoH')
        if soh_model_info:
            display_metrics_cards(soh_model_info, 'SoH')
            display_model_info(soh_model_info, 'SoH')
            
            all_soh_models = get_all_models_by_target('SoH')
            display_models_comparison(all_soh_models, 'SoH')
        else:
            st.warning("⚠️ No SoH models available yet.")

def show_soc_prediction():
    """Display SOC prediction page"""
    st.markdown("## 🔮 SOC Prediction")
    
    # Load SOC model
    soc_model, soc_model_info = load_model_from_config('SOC')
    
    # Display temperature range info
    if soc_model_info and 'dataset_info' in soc_model_info:
        dataset_info = soc_model_info['dataset_info']
        if 'temperature_range' in dataset_info:
            min_temp = dataset_info['temperature_range']['min']
            max_temp = dataset_info['temperature_range']['max']
            st.markdown(f'''
                <div style="background: transparent; border: 2px solid #1f77b4; border-radius: 15px; padding: 1.5rem; margin: 1rem 0;">
                    <h4 style="color: #1f77b4; margin: 0 0 0.5rem 0;">🌡️ Temperature Range in Dataset</h4>
                    <p style="color: black; font-weight: bold; margin: 0;"><strong>{min_temp:.2f}°C</strong> to <strong>{max_temp:.2f}°C</strong></p>
                </div>
            ''', unsafe_allow_html=True)
    
    # Prediction form
    inputs = create_prediction_form('SOC')
    
    if inputs and soc_model is not None:
        input_df = pd.DataFrame([inputs])
        soc_pred = soc_model.predict(input_df)[0]
        
        st.markdown("### 🎯 Prediction Result")
        display_prediction_result(soc_pred, 'SOC')
    elif inputs:
        st.error('❌ No SOC model available for prediction.')

def show_soh_prediction():
    """Display SoH prediction page"""
    st.markdown("## 🏥 SoH Prediction")
    
    # Load SoH model
    soh_model, soh_model_info = load_model_from_config('SoH')
    
    if soh_model_info:
        # Display model info
        st.info(f"**SoH Model:** {soh_model_info.get('model_name', 'N/A')} | **Performance:** R² = {soh_model_info.get('performance', {}).get('r2_score', 'N/A'):.4f}")
    
    # Prediction form
    inputs = create_prediction_form('SoH')
    
    if inputs and soh_model is not None:
        input_df = pd.DataFrame([inputs])
        soh_pred = soh_model.predict(input_df)[0]
        
        st.markdown("### 🎯 Prediction Result")
        display_prediction_result(soh_pred, 'SoH')
    elif inputs and soh_model is None:
        st.error('❌ No SoH model available for prediction.')
    elif not soh_model_info:
        st.warning("⚠️ SoH prediction is not yet configured. Please train SoH models first.")

def show_dataset_visualization():
    """Display dataset visualization page"""
    st.markdown("## 📊 Dataset Analysis & Visualization")
    
    dataset_info = get_dataset_info()
    if not dataset_info:
        st.warning('⚠️ Dataset information not available.')
        return
    
    # Dataset overview
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_records = dataset_info.get('total_records', 'N/A')
        st.metric("📊 Total Records", f"{total_records:,}" if isinstance(total_records, int) else str(total_records))
    with col2:
        soc_model_info = get_production_model_info('SOC')
        features_count = len(soc_model_info.get('features_used', [])) if soc_model_info else 'N/A'
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

# ==================== MAIN APPLICATION ====================

def main():
    """Main application function"""
    # Setup
    setup_page_config()
    apply_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🔋 Battery Management Dashboard</h1>', unsafe_allow_html=True)
    
    # Authentication
    if not check_authentication():
        login()
        st.stop()
    
    # Sidebar and navigation
    menu = create_sidebar()
    
    # Route to appropriate page
    if menu == '📈 Model Evaluation':
        show_model_evaluation()
    elif menu == '🔮 Predict SOC':
        show_soc_prediction()
    elif menu == '🏥 Predict SoH':
        show_soh_prediction()
    elif menu == '📊 Dataset Visualization':
        show_dataset_visualization()

if __name__ == "__main__":
    DATASET_PATH = 'unibo-powertools-dataset\\unibo-powertools-dataset\\test_result_trial_end_cleaned_v1.0.csv'  
    update_config_dataset_info(CONFIG_PATH, DATASET_PATH, target_col='SOC')
    update_config_dataset_info(CONFIG_PATH, DATASET_PATH, target_col='SoH')
    main()