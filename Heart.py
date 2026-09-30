import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("cardiovascular_disease_dataset.csv")
st.set_page_config(page_title="HeartSync AI",page_icon="icon.png",layout="wide")

with st.sidebar:
    selected = option_menu("HeartSync AI",
                           options=["Dashboard","Patient Information","Prediction","Health Report","Dataset","About Project"],
                           menu_icon=["heart-pulse-fill"],
                           icons=["speedometer2","person-vcard","heart-pulse","clipboard2-pulse","table","info-circle"])

if selected == "Dashboard":
    st.header("HeartSync AI")
    st.subheader("Predictive Cardiac Disease Detection Platform")
    st.write("""HeartSync AI is an AI-powered healthcare application that predicts the risk of cardiovascular 
    disease using Machine Learning.It analyzes important patient health parameters such as age, BMI, blood pressure, 
    cholesterol, blood sugar, heart rate, chest pain type, smoking status, exercise-induced angina, and family 
    history to determine whether a patient is at risk of cardiovascular disease.""")
    st.divider()

    total_patients = len(df)
    avg_age = round(df["Age"].mean(), 1)
    male_patients = len(df[df["Gender"] == "Male"])
    female_patients = len(df[df["Gender"] == "Female"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", total_patients)
    col2.metric("Average Age", avg_age)
    col3.metric("Male Patients", male_patients)
    col4.metric("Female Patients", female_patients)
    st.divider()

    avg_bmi = round(df["BMI"].mean(), 1)
    avg_bp = round(df["Systolic_BP"].mean(), 1)
    avg_cholesterol = round(df["Total_Cholesterol"].mean(), 1)
    heart_cases = len(df[df["Cardiovascular_Disease"] == "Yes"])

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Average BMI", avg_bmi)
    col6.metric("Average BP", avg_bp)
    col7.metric("Average Cholesterol", avg_cholesterol)
    col8.metric("Heart Disease Cases", heart_cases)
    st.divider()

    col9, col10 = st.columns(2)
    with col9:
        st.subheader("Heart Disease Distribution")
        disease = df["Cardiovascular_Disease"].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(disease.index,disease.values)
        ax.set_xlabel("Disease Status")
        ax.set_ylabel("Patients")
        st.pyplot(fig)
    with col10:
        st.subheader("Blood Pressure Category")
        bp = pd.cut(df["Systolic_BP"],bins=[0, 120, 140, 250],labels=["Normal", "Elevated", "High"])
        bp_count = bp.value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(bp_count.index, bp_count.values)
        ax.set_xlabel("Blood Pressure")
        ax.set_ylabel("Patients")
        st.pyplot(fig)
    st.divider()

    col11, col12 = st.columns(2)
    with col11:
        st.subheader("Chest Pain Type Analysis")
        chest = df["Chest_Pain_Type"].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(chest.index,chest.values)
        ax.set_xlabel("Chest Pain Type")
        ax.set_ylabel("Patients")
        plt.xticks(rotation=20)
        st.pyplot(fig)
    with col12:
        st.subheader("Smoking Status Analysis")
        smoke = df["Smoking_Status"].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        ax.pie(smoke.values,labels=smoke.index,autopct="%1.2f%%",startangle=90)
        st.pyplot(fig)

if selected == "Patient Information":
    st.header("Patient Information")
    st.subheader("Enter the patient's health details")

    col1, col2 = st.columns(2)
    with col1:
        name=st.text_input("Name ",value="ABC")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender",df["Gender"].unique())
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        systolic_bp = st.number_input("Systolic Blood Pressure",min_value=80,max_value=250,value=120)
        diastolic_bp = st.number_input("Diastolic Blood Pressure",min_value=40,max_value=150,value=80)
        total_cholesterol = st.number_input("Total Cholesterol",min_value=100,max_value=500,value=180)
        hdl = st.number_input("HDL Cholesterol",min_value=20,max_value=120,value=50)
    with col2:
        ldl = st.number_input("LDL Cholesterol", min_value=40, max_value=300, value=100)
        fasting_bs = st.number_input("Fasting Blood Sugar",min_value=50,max_value=300,value=90)
        resting_hr = st.number_input("Resting Heart Rate",min_value=40,max_value=150,value=72)
        max_hr = st.number_input("Maximum Heart Rate",min_value=60,max_value=220,value=150)
        chest_pain = st.selectbox("Chest Pain Type",df["Chest_Pain_Type"].unique())
        smoking = st.selectbox("Smoking Status",df["Smoking_Status"].unique())
        exercise = st.selectbox("Exercise Induced Angina",df["Exercise_Induced_Angina"].unique())
        family_history = st.selectbox("Family History of CVD",df["Family_History_CVD"].unique())

    if st.button("Save",use_container_width=True):
        st.session_state.patient_data = {
            "Name":name,
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "Systolic_BP": systolic_bp,
            "Diastolic_BP": diastolic_bp,
            "Total_Cholesterol": total_cholesterol,
            "HDL_Cholesterol": hdl,
            "LDL_Cholesterol": ldl,
            "Fasting_Blood_Sugar": fasting_bs,
            "Resting_Heart_Rate": resting_hr,
            "Max_Heart_Rate": max_hr,
            "Chest_Pain_Type": chest_pain,
            "Smoking_Status": smoking,
            "Exercise_Induced_Angina": exercise,
            "Family_History_CVD": family_history
        }
        st.toast("Patient information saved successfully.")

if selected == "Prediction":
    st.header("HeartSync AI Prediction")
    if "patient_data" not in st.session_state:
        st.warning("Please enter Patient Information first.")
        st.stop()
    patient = st.session_state.patient_data
    le_gender = LabelEncoder()
    le_chest = LabelEncoder()
    le_smoke = LabelEncoder()
    le_angina = LabelEncoder()
    le_family = LabelEncoder()
    le_target = LabelEncoder()

    df["Gender"] = le_gender.fit_transform(df["Gender"])
    df["Chest_Pain_Type"] = le_chest.fit_transform(df["Chest_Pain_Type"])
    df["Smoking_Status"] = le_smoke.fit_transform(df["Smoking_Status"])
    df["Exercise_Induced_Angina"] = le_angina.fit_transform(df["Exercise_Induced_Angina"])
    df["Family_History_CVD"] = le_family.fit_transform(df["Family_History_CVD"])
    df["Cardiovascular_Disease"] = le_target.fit_transform(df["Cardiovascular_Disease"])

    x = df[["Age","Gender","BMI","Systolic_BP","Diastolic_BP","Total_Cholesterol","HDL_Cholesterol","LDL_Cholesterol","Fasting_Blood_Sugar","Resting_Heart_Rate","Max_Heart_Rate","Chest_Pain_Type","Smoking_Status","Exercise_Induced_Angina","Family_History_CVD"]]
    y = df["Cardiovascular_Disease"]

    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.03,random_state=42)
    model = RandomForestClassifier(n_estimators=200,random_state=42)
    model.fit(xtrain, ytrain)
    prediction = model.predict(xtest)
    accuracy = accuracy_score(ytest, prediction)

    st.subheader("Patient Details")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name :** {patient['Name']}")
        st.write(f"**Age :** {patient['Age']}")
        st.write(f"**Gender :** {patient['Gender']}")
        st.write(f"**BMI :** {patient['BMI']}")
        st.write(f"**Systolic BP :** {patient['Systolic_BP']}")
        st.write(f"**Diastolic BP :** {patient['Diastolic_BP']}")
        st.write(f"**Total Cholesterol :** {patient['Total_Cholesterol']}")
        st.write(f"**HDL Cholesterol :** {patient['HDL_Cholesterol']}")
    with col2:
        st.write(f"**LDL Cholesterol :** {patient['LDL_Cholesterol']}")
        st.write(f"**Blood Sugar :** {patient['Fasting_Blood_Sugar']}")
        st.write(f"**Resting Heart Rate :** {patient['Resting_Heart_Rate']}")
        st.write(f"**Maximum Heart Rate :** {patient['Max_Heart_Rate']}")
        st.write(f"**Chest Pain Type :** {patient['Chest_Pain_Type']}")
        st.write(f"**Smoking Status :** {patient['Smoking_Status']}")
        st.write(f"**Exercise Angina :** {patient['Exercise_Induced_Angina']}")
        st.write(f"**Family History :** {patient['Family_History_CVD']}")

    if st.button("Predict Heart Disease", use_container_width=True):
        new_data = [[patient["Age"],le_gender.transform([patient["Gender"]])[0],
            patient["BMI"],patient["Systolic_BP"],patient["Diastolic_BP"],
            patient["Total_Cholesterol"],patient["HDL_Cholesterol"],patient["LDL_Cholesterol"],
            patient["Fasting_Blood_Sugar"],patient["Resting_Heart_Rate"],patient["Max_Heart_Rate"],
            le_chest.transform([patient["Chest_Pain_Type"]])[0],
            le_smoke.transform([patient["Smoking_Status"]])[0],
            le_angina.transform([patient["Exercise_Induced_Angina"]])[0],
            le_family.transform([patient["Family_History_CVD"]])[0]
        ]]
        result = model.predict(new_data)
        st.session_state.prediction_result = result[0]
        st.session_state.model_accuracy = accuracy

        st.subheader("Prediction Result")
        if result[0] == 1:
            st.error("High Risk of Cardiovascular Disease")
            st.snow()
        else:
            st.success("Low Risk of Cardiovascular Disease")
            st.balloons()
        st.success(f"Model Accuracy :{accuracy:.2%}")

if selected == "Health Report":
    st.header("Health Report")
    if "patient_data" not in st.session_state:
        st.warning("Please enter Patient Information first.")
        st.stop()
    if "prediction_result" not in st.session_state:
        st.warning("Please predict heart disease first.")
        st.stop()

    patient = st.session_state.patient_data
    result = st.session_state.prediction_result
    accuracy = st.session_state.model_accuracy

    st.subheader("Patient Details")
    report = pd.DataFrame({
        "Parameter": ["Name","Age","Gender","BMI","Systolic BP","Diastolic BP","Total Cholesterol",
            "HDL Cholesterol","LDL Cholesterol","Fasting Blood Sugar","Resting Heart Rate",
            "Maximum Heart Rate","Chest Pain Type","Smoking Status","Exercise Induced Angina",
            "Family History"],
        "Value": [patient["Name"],patient["Age"],patient["Gender"],patient["BMI"],patient["Systolic_BP"],
            patient["Diastolic_BP"],patient["Total_Cholesterol"],patient["HDL_Cholesterol"],
            patient["LDL_Cholesterol"],patient["Fasting_Blood_Sugar"],patient["Resting_Heart_Rate"],
            patient["Max_Heart_Rate"],patient["Chest_Pain_Type"],patient["Smoking_Status"],
            patient["Exercise_Induced_Angina"],patient["Family_History_CVD"]]
    })
    st.dataframe(report, use_container_width=True)
    st.divider()

    st.subheader("Result")
    if result == 1:
        st.error("High Risk of Cardiovascular Disease")
        st.warning("""### Health Recommendations
        • Consult a cardiologist immediately.
        • Monitor blood pressure regularly.
        • Reduce salt, sugar and fatty foods.
        • Avoid smoking and alcohol.
        • Exercise only as advised by your doctor.
        • Maintain a healthy body weight.
        • Take prescribed medicines regularly.""")
    else:
        st.success("Low Risk of Cardiovascular Disease")
        st.info("""### Health Recommendations
        • Continue regular exercise.
        • Eat a balanced diet.
        • Maintain healthy body weight.
        • Monitor blood pressure regularly.
        • Get routine health check-ups.
        • Drink enough water.
        • Sleep at least 7–8 hours daily.""")

if selected == "Dataset":
    st.header("Dataset Overview")
    st.subheader("Below is the dataset of air qualty predicition")
    st.dataframe(df)
    st.write(df.shape[0]," Rows *  ",df.shape[1]," Columns")
    st.divider()
    st.subheader("Dataset Summary")
    st.write(df.describe())

if selected == "About Project":

    st.header("About HeartSync AI")
    st.write("""HeartSync AI is a machine learning-based healthcare application designed to predict the risk of 
    cardiovascular disease using patient health data.The system analyzes important medical parameters such as 
    age, BMI, blood pressure, cholesterol levels, blood sugar, heart rate, smoking status, chest pain type, 
    exercise-induced angina, and family history to provide an early prediction of heart disease.""")
    st.divider()

    st.subheader("Project Objective")
    st.write("""The main objective of HeartSync AI is to assist in the early detection of cardiovascular disease by 
    applying machine learning techniques.Early prediction helps patients seek medical advice sooner and supports
    preventive healthcare.""")
    st.divider()

    st.subheader("Technology Used")
    st.write("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Random Forest Classifier
    """)
    st.divider()

    st.subheader("Machine Learning Algorithm")
    st.write("""**Random Forest Classifier** → Random Forest is an ensemble machine learning algorithm that combines 
    multiple decision trees to improve prediction accuracy and reduce overfitting. It provides reliable 
    classification for healthcare prediction problems.""")
    st.divider()

    st.subheader("Project Features")
    st.write("""
    ✔ Patient Information Management \n
    ✔ Heart Disease Risk Prediction \n
    ✔ Interactive Dashboard\n
    ✔ Health Report Generation\n
    ✔ Dataset Visualization\n
    ✔ Machine Learning-Based Prediction\n""")
    st.divider()

    st.subheader("Developer")
    st.write("""
    **Project Name:** HeartSync AI\n
    **Developed By:** Pushti Doshi\n
    **Course:** Bachelor of Engineering (Computer Engineering)\n
    **Semester:** 7\n
    **Technology:** Machine Learning using Random Forest Classifier\n
    """)
    st.success("Thank you for using HeartSync AI.")