# Automated ML Trainer

A streamlined machine learning tool for rapid dataset analysis and feature relevance assessment, designed for both data science practitioners and educational purposes.

##  Problem Statement

In the rapidly evolving field of data science, researchers and analysts often face significant time constraints when conducting preliminary dataset analysis. The initial exploratory phase, which involves identifying relevant features and understanding their predictive power, can be both time-consuming and technically demanding. This challenge is particularly pronounced for:

- Data scientists who need to quickly assess multiple datasets for viability
- Researchers seeking to establish baseline relationships before investing in comprehensive analysis
- Educational institutions teaching machine learning concepts to students without programming backgrounds
- Business analysts who require rapid insights but lack extensive ML expertise

The Automated ML Trainer addresses these challenges by providing an intuitive, efficient solution for preliminary dataset analysis. This tool automates the process of feature relevance assessment, enabling users to quickly identify which variables demonstrate the strongest predictive relationships with their target outcomes. By streamlining this critical first step in the machine learning pipeline, the application significantly reduces the time investment required for initial data exploration while democratizing access to machine learning capabilities for non-technical users.

The platform serves a dual purpose: as a practical analytical tool for experienced practitioners seeking to optimize their workflow, and as an educational resource that demonstrates fundamental machine learning concepts through hands-on interaction, making it particularly valuable for academic settings where students can observe the ML training process without writing code.

##  Features

- **Automated Data Processing**: Handles CSV uploads with automatic data type detection
- **Smart Feature Selection**: Single-feature analysis for clear interpretability
- **Multiple Model Comparison**: Automatically trains and compares multiple ML algorithms
- **Adaptive Model Selection**: Chooses between regression and classification based on target variable
- **Real-time Predictions**: Interactive prediction interface with custom inputs
- **No-code Solution**: Complete ML pipeline accessible through intuitive UI

##  Technical Implementation

### Core Architecture

The application is built using Streamlit and structured into six main containers:

1. **Header Container**: Application introduction and overview
2. **Dataset Container**: CSV file upload and data preprocessing
3. **Feature Selection Container**: Target variable and feature selection interface
4. **Model Training Container**: Automated model training with multiple algorithms
5. **Model Performance Container**: Model evaluation and best model selection
6. **Predictions Container**: Interactive prediction interface

### Key Components

#### Data Processing
```python
@st.cache_data
def get_dataset(filepath):
    data = pd.read_csv(filepath)
    return(data)
```

- Utilizes Streamlit's caching for optimized performance
- Automatic identification and handling of non-numeric columns
- Data validation and error handling

#### Model Selection Logic

For regression tasks involving continuous targets, the application employs models such as Random Forest Regressor, Decision Tree Regressor, and Linear Regression. In classification scenarios where targets are categorical, it utilizes Random Forest Classifier, Logistic Regression, and Naïve Bayes models.

#### Evaluation Metrics
The evaluation metrics are tailored to the problem type: Mean Absolute Error (MAE) and Mean Squared Error (MSE) are used for regression tasks, while Accuracy Score is applied for classification tasks.

## Example Use Case: Student Performance Analysis

### Dataset Overview
A student performance dataset containing:
- **Subject**: Academic subject (Language, Science, Maths)
- **Marks**: Student scores (35.4 - 97)
- **Hours**: Study hours (1 - 7.7 hours)
- **Student_id**: Unique identifier
- **Year**: Academic year

### Analysis Workflow

1. **Data Upload**: CSV file uploaded through the interface
2. **Target Selection**: "Marks" selected as target variable (regression task)
3. **Feature Analysis**: "Hours" selected to analyze study time impact
4. **Model Training**: Three regression models trained automatically
5. **Results**: 
   - Random Forest Regressor achieved best performance (lowest MSE)
   - Clear positive correlation between study hours and marks identified
   - Model successfully predicts marks based on study hours input

### Key Insights
- Study hours show a strong predictive relationship with student marks.

- The strength of this relationship varies based on the amount of time spent studying.

- Marks tend to increase as study hours increase, up to a certain point.

- Beyond this optimal point, performance begins to slightly decline, suggesting that excessive studying may have a negative effect—similar to studying too little.

- The tool used was able to identify these meaningful patterns effectively, without requiring manual coding.

## Installation & Usage

### Prerequisites
``` BASH

pip install streamlit pandas numpy scikit-learn
```

### Running Locally
```BASH

streamlit run app.py
```

### Usage Steps

**Upload your CSV dataset**

- Review data and drop non-numeric columns if needed
- Select target variable and specify if it's categorical
- Choose a feature for analysis
- Train models with one click
- View performance comparison
- Make predictions with custom inputs
  
## Live Demo
Try the Automated ML Trainer online: [Live Demo Link] (To be provided)



