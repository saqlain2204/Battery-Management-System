# 🔋 Battery Management System - SOC Prediction Dashboard

A comprehensive battery State of Charge (SOC) prediction system using machine learning models with an interactive Streamlit dashboard.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [MATLAB Integration](#matlab-integration)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Dataset](#dataset)
- [Authentication](#authentication)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides a complete battery management solution that:
- Predicts State of Charge (SOC) using machine learning models
- Offers an interactive web dashboard for visualization and prediction
- Supports multiple ML algorithms (Random Forest, Linear Regression, Decision Tree, Gradient Boosting)
- Provides MATLAB-compatible model exports
- Includes comprehensive data analysis and visualization tools

## ✨ Features

### 🔬 Machine Learning Models
- **Random Forest Regressor** - Best performance (R² ≈ 0.997)
- **Decision Tree Regressor** - Good interpretability
- **Linear Regression** - Simple baseline model
- **Gradient Boosting Regressor** - Advanced ensemble method

### 📊 Interactive Dashboard
- **Model Evaluation**: View performance metrics and comparisons
- **SOC Prediction**: Real-time prediction with dynamic color coding
- **Data Visualization**: Interactive plots and correlation analysis
- **Authentication**: Secure login system

### 🔧 MATLAB Compatibility
- Model parameter extraction
- MATLAB prediction scripts
- Batch prediction support
- Complete documentation

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git (optional)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd bms
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies List
```
streamlit
pandas
scikit-learn
joblib
numpy
seaborn
matplotlib
plotly
statsmodels
scipy
```

## 🏃 Quick Start

### 1. Prepare Dataset
Ensure you have the cleaned dataset:
```
unibo-powertools-dataset/unibo-powertools-dataset/test_result_trial_end_cleaned_v1.0.csv
```

### 2. Train Models (Optional)
If you want to retrain models:
```bash
# Open Jupyter notebook
jupyter notebook experiments/training_models.ipynb
# Run all cells to train and save models
```

### 3. Run Dashboard
```bash
streamlit run app.py
```

### 4. Access Application
- Open browser to `http://localhost:8501`
- **Login Credentials:**
  - Username: `admin`
  - Password: `battery123`

## 📖 Usage

### Dashboard Navigation

#### 1. 📈 Model Evaluation
- View model performance metrics (RMSE, R² Score, MAE)
- Compare different models
- Analyze training statistics
- Interactive model comparison charts

#### 2. 🔮 Predict SOC
- Enter battery parameters:
  - **Time**: Time value for prediction
  - **Voltage**: Battery voltage (0-5V)
  - **Current**: Battery current (-10 to 10A)
  - **Temperature**: Max temperature (-10°C to 47°C)
- Get real-time SOC prediction with color-coded status
- Battery health interpretation

#### 3. 📊 Dataset Visualization
- Interactive temperature distribution
- Correlation heatmaps
- Feature relationship plots
- Traditional pairplot analysis

### Command Line Tools

#### Convert Models to MATLAB
```bash
cd scripts
python convert_models_to_matlab.py
```

#### Run EDA Notebook
```bash
jupyter notebook eda.ipynb
```

## 🔧 MATLAB Integration

### Converting Models

1. **Run Conversion Script:**
```bash
cd scripts
python convert_models_to_matlab.py
```

2. **Generated Files:**
- `models/matlab_compatible/*.mat` - MATLAB data files
- `models/matlab_compatible/*.json` - Human-readable parameters
- `models/matlab_compatible/*_features.json` - Feature information
- `models/matlab_compatible/soc_prediction_matlab.m` - Prediction script

### Using Models in MATLAB

#### Basic Prediction Example:
```matlab
%% Load Model
model_data = load('SOC_RandomForestRegressor_v1.0_test_result_trial_end_v1.0_20250804_003236.mat');

%% Define Input [time, voltage, current, max_temperature]
input_features = [0.0, 3.7, 0.0, 25.0];

%% Make Prediction
soc_prediction = predict_random_forest(input_features, model_data);
fprintf('Predicted SOC: %.2f%%\n', soc_prediction);
```

#### Batch Prediction:
```matlab
% Multiple samples
features_matrix = [
    0.0, 3.7, 0.0, 25.0;
    1.0, 3.6, -0.5, 30.0;
    2.0, 3.5, -1.0, 35.0
];

predictions = predict_batch(features_matrix, model_data);
```

#### Supported Model Types:
- **Linear Regression**: Direct coefficient multiplication
- **Random Forest**: Tree ensemble averaging
- **Decision Tree**: Single tree traversal
- **Gradient Boosting**: Weighted tree combination

### MATLAB Model Structure

#### Linear Model:
```matlab
model_data.coefficients  % Feature coefficients
model_data.intercept     % Model intercept
model_data.n_features    % Number of input features
```

#### Tree-based Models:
```matlab
model_data.trees         % Tree structures
model_data.n_estimators  % Number of trees
model_data.feature_importances % Feature importance scores
```

## 📁 Project Structure

```
bms/
├── 📊 app.py                          # Main Streamlit dashboard
├── 📓 eda.ipynb                       # Exploratory data analysis
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # This file
├── 🔒 .streamlit/
│   └── secrets.toml                   # Authentication credentials
├── 📂 experiments/
│   └── training_models.ipynb          # Model training notebook
├── 🤖 models/
│   ├── *.joblib                       # Trained Python models
│   └── matlab_compatible/             # MATLAB-compatible exports
│       ├── *.mat                      # MATLAB data files
│       ├── *.json                     # Parameter files
│       └── soc_prediction_matlab.m    # MATLAB prediction script
├── 📊 results/
│   ├── model_results.csv              # Model performance metrics
│   └── feature_importance_*.csv       # Feature importance data
├── 🔧 scripts/
│   └── convert_models_to_matlab.py    # Model conversion utility
└── 📁 unibo-powertools-dataset/
    └── unibo-powertools-dataset/
        └── test_result_trial_end_cleaned_v1.0.csv  # Clean dataset
```

## 🤖 Model Information

### Performance Comparison

| Model | R² Score | RMSE | MAE | Best Use Case |
|-------|----------|------|-----|---------------|
| **Random Forest** | 0.9968 | 1.0714 | 0.2397 | **Production** (Best overall) |
| **Decision Tree** | 0.9948 | 1.3635 | 0.2473 | Interpretability |
| **Gradient Boosting** | 0.9480 | 4.3139 | 3.2723 | Feature engineering |
| **Linear Regression** | 0.8599 | 7.0773 | 5.4789 | Baseline/Simple cases |

### Feature Importance (Random Forest)
1. **Voltage** (89.4%) - Primary SOC indicator
2. **Current** (4.8%) - Charge/discharge state
3. **Time** (4.8%) - Temporal patterns
4. **Temperature** (1.1%) - Environmental factor

### Input Features
- **Time**: Temporal sequence value
- **Voltage**: Battery terminal voltage (V)
- **Current**: Charge/discharge current (A)
- **Max Temperature**: Maximum recorded temperature (°C)

### Output
- **SOC**: State of Charge percentage (0-100%)

## 📊 Dataset

### Source
- **Origin**: UNIBO PowerTools Dataset
- **Size**: 405,765 samples (cleaned)
- **Features**: 5 (after cleaning)
- **Target**: State of Charge (SOC)

### Data Processing
1. **Temperature Filtering**: -10°C to 47°C range
2. **Missing Values**: Removed all NaN entries
3. **Duplicates**: Removed duplicate records
4. **SOC Calculation**: Based on charging/discharging capacity
5. **Negative SOC**: Removed invalid negative values

### Feature Ranges
- **Time**: 0 to ∞
- **Voltage**: 0 to 5V (typical: 3.0-4.2V)
- **Current**: -10 to 10A
- **Temperature**: -10°C to 47°C

## 🔐 Authentication

### Default Credentials
- **Username**: `admin`
- **Password**: `battery123`

### Customization
Edit `.streamlit/secrets.toml`:
```toml
[auth]
username = "your_username"
password = "your_password"
```

### Security Features
- Session-based authentication
- Logout functionality
- Credentials stored in secrets file
- Automatic login persistence

## 🛠️ Development

### Adding New Models

1. **Train Model** in `experiments/training_models.ipynb`
2. **Save Model** using the results tracking system
3. **Convert to MATLAB** using the conversion script
4. **Update Dashboard** if needed

### Custom Feature Engineering

Modify the feature selection in:
- `experiments/training_models.ipynb` (training)
- `app.py` (prediction interface)
- `scripts/convert_models_to_matlab.py` (MATLAB export)

### Extending MATLAB Support

Add new model types in `convert_models_to_matlab.py`:
```python
def convert_new_model_type(model, model_name, output_dir):
    # Extract parameters
    # Save as .mat and .json
    pass
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Install missing packages
pip install package_name
```

#### 2. Model Not Found
```bash
# Check models directory
ls models/
# Retrain if needed
jupyter notebook experiments/training_models.ipynb
```

#### 3. Dataset Issues
- Ensure cleaned dataset exists
- Check file path in configuration
- Verify data format

#### 4. MATLAB Conversion
```bash
# Install scipy for .mat file support
pip install scipy
```

#### 5. Streamlit Issues
```bash
# Clear cache
streamlit cache clear
# Restart application
streamlit run app.py
```

### Performance Optimization

#### Large Datasets
- Use data sampling in visualizations
- Implement data chunking for training
- Consider feature selection

#### Memory Usage
- Close matplotlib figures after use
- Use data generators for large files
- Monitor memory with system tools

## 📈 Future Enhancements

### Planned Features
- [ ] Real-time data streaming
- [ ] Advanced battery health diagnostics
- [ ] Multi-battery system support
- [ ] Cloud deployment options
- [ ] API endpoints for integration
- [ ] Advanced anomaly detection
- [ ] Battery degradation modeling

### MATLAB Enhancements
- [ ] Simulink block generation
- [ ] Real-time workshop integration
- [ ] Advanced visualization tools
- [ ] Parameter optimization functions

## 🤝 Contributing

### Getting Started
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Code Standards
- Follow PEP 8 for Python code
- Add docstrings to functions
- Include unit tests for new features
- Update documentation

### Reporting Issues
- Use GitHub Issues
- Include error messages
- Provide steps to reproduce
- Specify environment details

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UNIBO PowerTools Dataset** for providing battery data
- **Scikit-learn** for machine learning algorithms
- **Streamlit** for the interactive dashboard framework
- **Plotly** for interactive visualizations
- **MATLAB** community for compatibility requirements

## 📞 Support

For support and questions:
- 📧 Email: [your-email@domain.com]
- 🐛 Issues: GitHub Issues
- 📖 Documentation: This README
- 💬 Discussions: GitHub Discussions

---

**Made with ❤️ for Battery Management Systems**

*Last updated: August 4, 2025*
