# Student Exam Performance Prediction

An end-to-end machine learning project that predicts student math exam scores based on demographic and academic attributes. This project demonstrates the complete ML pipeline from data exploration, model training with hyperparameter tuning, to production deployment via a Flask web application.

**[View Live Demo](#quick-start)** | **[Architecture](#architecture)** | **[Models](#models-evaluated)** | **[Results](#results)**

---

## Features

- **Exploratory Data Analysis**: Comprehensive EDA notebook revealing patterns in student demographics and performance
- **Multiple ML Models**: Trains and compares 6 regression algorithms with automated hyperparameter tuning
- **Production Pipeline**: Complete data preprocessing with feature engineering (OneHotEncoding, StandardScaling)
- **Web Interface**: Flask application with an intuitive form for real-time predictions
- **Model Evaluation**: R-squared score-based model selection with performance metrics
- **Error Handling**: Custom exception tracking with detailed logging
- **Reproducible**: Setup script for easy environment configuration

---

## Project Overview

### Problem Statement
Predict a student's math exam score given their:
- **Demographics**: Gender, race/ethnicity, parental education level
- **Socioeconomic factors**: Lunch type (free/reduced vs. standard)
- **Academic prep**: Test preparation course completion
- **Prior performance**: Reading and writing scores

### Dataset
- **Source**: `notebook/data/stud.csv`
- **Target Variable**: `math_score`
- **Features**: 7 input features (5 categorical, 2 numerical)
- **Train-Test Split**: 80-20

---

## Architecture

### Project Structure

```
dsproject/
├── src/
│   ├── components/              # ML pipeline components
│   │   ├── data_ingestion.py    # Load & split raw data
│   │   ├── data_transformation.py # Feature engineering & preprocessing
│   │   └── model_trainer.py      # Model training & hyperparameter tuning
│   ├── pipeline/
│   │   ├── train_pipeline.py    # Full training orchestration
│   │   └── predict_pipeline.py  # Inference interface
│   ├── exception.py             # Custom exception handling with stack trace
│   ├── logger.py                # Logging configuration
│   └── utils.py                 # Helper functions (save/load models, GridSearch)
│
├── notebook/
│   ├── 1_eda_student_performance.ipynb  # Data exploration & visualization
│   ├── 2_model_training.ipynb            # Model experiments & tuning
│   └── data/stud.csv                     # Raw dataset
│
├── artifacts/                   # Trained model & preprocessor (generated)
│   ├── model.pkl               # Best trained regressor
│   └── preprocessor.pkl        # Feature transformation pipeline
│
├── templates/
│   └── index.html              # Flask web UI
├── static/                     # CSS & static assets
├── application.py              # Flask app entry point
├── setup.py                    # Package metadata
├── requirements.txt            # Python dependencies
└── README.md
```

### Data Flow

```
Raw Data (stud.csv)
    |
    v
[Data Ingestion] -> Train/Test Split (80/20)
    |
    v
[Data Transformation]
    ├─ Numerical Features -> SimpleImputer (median) -> StandardScaler
    └─ Categorical Features -> SimpleImputer (most_frequent) -> OneHotEncoder -> StandardScaler
    |
    v
[Model Trainer] -> GridSearchCV Hyperparameter Tuning -> Best Model Selection
    |
    v
[Artifacts Storage] (model.pkl, preprocessor.pkl)
    |
    v
[Flask Web App] <- User Input Form
    |
    v
[Predict Pipeline] -> Load Model/Preprocessor -> Transform Input -> Predict
    |
    v
Math Score Prediction
```

---

## Models Evaluated

The project compares the following regression algorithms using **3-fold cross-validation**:

| Model | Hyperparameters Tuned | Best Config |
|-------|----------------------|-------------|
| **Random Forest** | `n_estimators` | [8, 16, 32, 64, 128, 256] |
| **Decision Tree** | `criterion` | [squared_error, friedman_mse, absolute_error, poisson] |
| **Gradient Boosting** | `learning_rate`, `n_estimators` | lr in [0.1, 0.01, 0.05, 0.001], n_est in [8-256] |
| **Linear Regression** | None | Baseline |
| **K-Neighbors Regressor** | `n_neighbors` | [5, 7, 9, 11] |
| **AdaBoost Regressor** | `learning_rate`, `n_estimators` | lr in [0.1, 0.01, 0.5, 0.001], n_est in [8-256] |

**Selection Criterion**: Highest R-squared score on test set (threshold: R-squared > 0.60)

---

## Results

The trained model achieves strong predictive performance on the test set. Model metrics are displayed in `application.py` after training completes.

**Key Findings from EDA** (see `1_eda_student_performance.ipynb`):
- Reading and writing scores are strong predictors of math performance
- Parental education level correlates positively with student outcomes
- Test preparation course completion shows measurable impact

---

## Quick Start

### Prerequisites
- Python 3.11
- pip or conda

### Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/erickgathinji/dsproject.git
cd dsproject

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in development mode
pip install -e .
```

### Training the Model

```bash
# Run the complete training pipeline
# This will:
# - Load the raw dataset
# - Perform train-test split
# - Apply feature transformations
# - Train all models with hyperparameter tuning
# - Save best model & preprocessor to artifacts/
python src/components/data_ingestion.py
```

**Expected Output**:
```
Entered the data ingestion method or component
Read the dataset as dataframe
Train test split initiated
Data ingestion is completed
[Training progress...]
Best model found! Random Forest
Model saved!
```

### Running the Flask Application

```bash
# Start the web server
python application.py

# Visit the root endpoint to access the prediction form: http://localhost:5000
```

The app will:
1. Display an interactive form to input student characteristics
2. Send POST request to `/` endpoint with form data
3. Load the trained model and preprocessor from `artifacts/`
4. Return predicted math score on the same page


## Data Preprocessing Details

### Numerical Features
- **Columns**: `reading_score`, `writing_score`
- **Pipeline**:
  1. `SimpleImputer(strategy='median')` - Handle missing values
  2. `StandardScaler()` - Normalize to mean=0, std=1

### Categorical Features
- **Columns**: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`
- **Pipeline**:
  1. `SimpleImputer(strategy='most_frequent')` - Fill with mode
  2. `OneHotEncoder()` - Convert to binary features
  3. `StandardScaler(with_mean=False)` - Scale (sparse-safe)

All preprocessing is fitted on training data and applied identically to test and production data via serialized `preprocessor.pkl`.

---

## Logging & Error Handling

The project implements robust error tracking:

### Custom Exception
- Captures exception type, file name, line number, and error message
- Provides detailed traceback for debugging
- File: `src/exception.py`

### Logging
- Logs all major pipeline steps (data loading, transformations, training)
- Both console and file output
- File: `src/logger.py`

### Flask Error Handling
- Returns 500 status with formatted error details
- Displays full traceback in browser for development

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.2.6 | Numerical computing |
| scikit-learn | 1.7.2 | ML algorithms & preprocessing |
| Flask | 3.1.3 | Web framework |
| dill | 0.4.1 | Model serialization |
| matplotlib | 3.10.9 | Visualization |
| seaborn | 0.13.2 | Statistical visualization |

---

## Notebooks

### 1. EDA Notebook (`1_eda_student_performance.ipynb`)
- Dataset overview and statistics
- Missing value analysis
- Distribution plots for numerical features
- Categorical feature analysis
- Correlation heatmaps
- Insights on relationships between features and target

### 2. Model Training Notebook (`2_model_training.ipynb`)
- Feature preprocessing demonstration
- Model training and evaluation
- Hyperparameter tuning experiments
- Performance comparison charts

---

## Deployment

The application is deployed and running on AWS Elastic Beanstalk. You can access the live environment here:

[Live Project Application](http://studentperformance-env.eba-6c78fg3i.us-east-1.elasticbeanstalk.com/)

This project is made with care for educational purposes.
