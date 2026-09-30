# ❤️ HeartSync AI

> **AI-powered cardiovascular disease risk prediction platform built
> with Python, Streamlit, and Random Forest.**

## 📌 Overview

**HeartSync AI** is a machine-learning-based healthcare application
designed to predict the risk of cardiovascular disease using patient
health information.

The application provides an interactive Streamlit interface where users
can enter patient health parameters and receive a cardiovascular disease
risk prediction. It also includes an interactive dashboard, patient
information management, dataset visualization, and a health report
section.

The machine learning model uses a **Random Forest Classifier** to
analyze multiple patient-related features and classify cardiovascular
disease risk.

## ✨ Features

-   ❤️ Cardiovascular disease risk prediction
-   👤 Patient information management
-   📊 Interactive healthcare dashboard
-   🩺 Blood pressure analysis
-   🧪 Cholesterol and BMI analysis
-   💓 Heart-rate analysis
-   🚬 Smoking-status analysis
-   🫀 Chest-pain analysis
-   🤖 Random Forest machine learning model
-   📋 Patient health report
-   📈 Dataset visualization and summary
-   ℹ️ Project information section
-   🎨 Custom application icon and Streamlit interface

## 🧠 Machine Learning

HeartSync AI uses the **Random Forest Classifier** for cardiovascular
disease prediction.

The model analyzes the following patient parameters:

-   Age
-   Gender
-   BMI
-   Systolic Blood Pressure
-   Diastolic Blood Pressure
-   Total Cholesterol
-   HDL Cholesterol
-   LDL Cholesterol
-   Fasting Blood Sugar
-   Resting Heart Rate
-   Maximum Heart Rate
-   Chest Pain Type
-   Smoking Status
-   Exercise-Induced Angina
-   Family History of Cardiovascular Disease

Categorical values are converted into numerical values using
`LabelEncoder`, and the dataset is divided into training and testing
data before training the Random Forest model.

## 🖥️ Application Modules

### 1. Dashboard

The dashboard provides an overview of the dataset and displays:

-   Total patients
-   Average age
-   Male and female patient counts
-   Average BMI
-   Average blood pressure
-   Average cholesterol
-   Cardiovascular disease cases
-   Heart disease distribution
-   Blood pressure categories
-   Chest pain analysis
-   Smoking status analysis

### 2. Patient Information

Users can enter patient information such as:

-   Name
-   Age
-   Gender
-   BMI
-   Blood pressure
-   Cholesterol levels
-   Blood sugar
-   Heart rate
-   Chest pain type
-   Smoking status
-   Exercise-induced angina
-   Family history of cardiovascular disease

### 3. Prediction

After entering patient information, the application uses the trained
Random Forest model to generate a cardiovascular disease risk
prediction.

The application displays:

-   Patient details
-   Prediction result
-   Model accuracy

### 4. Health Report

The health report displays the patient's information and prediction
result along with general health recommendations based on the
prediction.

### 5. Dataset

The Dataset section provides:

-   Complete dataset visualization
-   Number of rows and columns
-   Statistical summary of the dataset

### 6. About Project

This section describes the project objective, technologies used, machine
learning algorithm, project features, and developer information.

## 🛠️ Technologies Used

  Technology      Purpose
  --------------- ---------------------------------------
  Python          Core programming language
  Streamlit       Web application and user interface
  Pandas          Data loading and analysis
  Matplotlib      Data visualization
  Scikit-learn    Machine learning
  Random Forest   Cardiovascular disease classification
  LabelEncoder    Encoding categorical data

## 📂 Project Structure

``` text
HeartSync-AI/
│
├── 1.py
├── cardiovascular_disease_dataset.csv
├── icon.png
└── README.md
```

### File Description

-   **`1.py`** --- Main Streamlit application containing the dashboard,
    patient input, prediction, health report, dataset, and project
    information modules.
-   **`cardiovascular_disease_dataset.csv`** --- Dataset used by the
    application for analysis and machine learning.
-   **`icon.png`** --- Application icon.
-   **`README.md`** --- Project documentation.

## 🚀 Installation

### 1. Clone the repository

``` bash
git clone https://github.com/KARANVAGHELA423/HeartSync-AI.git
```

### 2. Open the project folder

``` bash
cd HeartSync-AI
```

### 3. Install required libraries

``` bash
pip install pandas matplotlib streamlit streamlit-option-menu scikit-learn
```

### 4. Run the application

``` bash
streamlit run 1.py
```

The application will open in your browser.

## 🔄 How It Works

``` text
Patient Health Information
          ↓
     Data Processing
          ↓
   Categorical Encoding
          ↓
     Random Forest
          ↓
   Risk Classification
          ↓
  Health Report & Result
```

## 📊 Dataset

The project uses a cardiovascular disease dataset containing patient
health information and a cardiovascular disease target variable.

The application uses the dataset for:

-   Exploratory analysis
-   Dashboard statistics
-   Data visualization
-   Machine learning model training
-   Cardiovascular disease prediction

## 🎯 Project Objective

The main objective of HeartSync AI is to demonstrate how machine
learning can be applied to healthcare data to assist with the early
identification of cardiovascular disease risk.

The application provides an easy-to-use interface for exploring patient
data and demonstrating machine-learning-based classification.

## ⚠️ Disclaimer

**HeartSync AI is an educational and demonstration project.**

Its predictions and recommendations should **not** be considered a
medical diagnosis or a replacement for professional medical advice. Real
healthcare decisions should always be made by qualified healthcare
professionals using appropriate clinical evaluation and medical tests.

## 👨‍💻 Developer

**KARAN**

Bachelor of Engineering (Computer Engineering)\
Machine Learning using Random Forest Classifier

## 📜 License

This project is intended for educational and portfolio purposes.
