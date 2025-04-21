import pickle
import numpy as np
import streamlit as st
import sqlite3
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Multiple Disease Prediction", layout="centered")

# ------------------ MODEL LOADING ------------------
with open('diabetes_model.sav', 'rb') as model_file:
    diabetes_model = pickle.load(model_file)
with open('scaler.sav', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open('parkinsons_model.sav', 'rb') as model_file:
    parkinsons_model = pickle.load(model_file)
with open('parkinsons_scaler.sav', 'rb') as scaler_file:
    parkinsons_scaler = pickle.load(scaler_file)

# ------------------ DATABASE SETUP ------------------
conn = sqlite3.connect('disease_predictions.db', check_same_thread=False)
c = conn.cursor()

# Create tables if not exist
c.execute('''
    CREATE TABLE IF NOT EXISTS diabetes_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pregnancies INTEGER,
        glucose INTEGER,
        blood_pressure INTEGER,
        skin_thickness INTEGER,
        insulin INTEGER,
        bmi REAL,
        dpf REAL,
        age INTEGER,
        result TEXT
    )
''')

# Ensure the table is created correctly (without dropping)
c.execute(''' 
    CREATE TABLE IF NOT EXISTS parkinsons_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_data TEXT,
        result TEXT
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS heart_disease_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        sex INTEGER,
        cp INTEGER,
        trestbps INTEGER,
        chol INTEGER,
        fbs INTEGER,
        restecg INTEGER,
        thalach INTEGER,
        exang INTEGER,
        oldpeak REAL,
        slope INTEGER,
        ca INTEGER,
        thal INTEGER,
        result TEXT
    )
''')



conn.commit()

# ------------------ SIDEBAR MENU ------------------
with st.sidebar:
    selected = option_menu(' Multiple Disease Prediction System', 
                           ['Diabetes Prediction', "Parkinson's Prediction","Heart Disease Prediction", 'View History'], 
                           icons=['activity', 'person', 'heart-pulse', 'search',], 
                           default_index=0)

# ------------------ DIABETES SECTION ------------------
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction')

    # Description and image for Diabetes
    st.markdown("""
    ### Diabetes Disease
    Diabetes is a chronic condition that affects how the body processes blood sugar (glucose).
    There are two main types of diabetes:
    - Type 1 Diabetes: The body doesn’t produce insulin.
    - Type 2 Diabetes: The body doesn’t respond to insulin properly.

    Common symptoms include:
    - Increased thirst
    - Frequent urination
    - Extreme hunger
    - Unexplained weight loss
    - Fatigue
    - Blurred vision

    ![Diabetes Image](https://www.medicoverhospitals.in/images/diseases/diabetes-complication.webp) 
                
    The following factors are considered for Diabetes prediction:
""")

    Pregnancies = st.text_input("Number of Pregnancies")
    SkinThickness = st.text_input('Skin Thickness Value')
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Glucose = st.text_input('Glucose Level')
    Insulin = st.text_input('Insulin Level')
    Age = st.text_input('Age of the person')
    BloodPressure = st.text_input('Blood Pressure Value')
    BMI = st.text_input('BMI Value')


    if st.button("Diabetes Prediction Result"):
            try:
                input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]).astype(float)
                scaled_data = loaded_scaler.transform(input_data)
                prediction = diabetes_model.predict(scaled_data)

                result = 'The person is Diabetic' if prediction[0] == 1 else 'The person is not Diabetic'
                st.success(result)

                # Save to DB
                c.execute(''' 
                    INSERT INTO diabetes_predictions (
                        pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, result
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, result))
                conn.commit()

            except ValueError as e:
                st.error(f"Error in input data: {e}")

#----------------- HEART DISEASE SECTION ------------------
elif selected == "Heart Disease Prediction":
    # Load model and scaler
    with open('heart_disease_model.sav', 'rb') as model_file:
        model = pickle.load(model_file)

    st.title("🫀 Heart Disease Prediction")
      # Description about heart disease
    st.markdown("""
    ## About Heart Disease

    Heart disease, also known as cardiovascular disease (CVD), refers to a group of diseases that affect the heart and blood vessels. It is one of the leading causes of death worldwide. Common types of heart disease include coronary artery disease (CAD), heart attacks, heart failure, and arrhythmias (irregular heartbeats).

    The risk factors for heart disease include high blood pressure, high cholesterol, smoking, diabetes, obesity, and a lack of physical activity. A healthy lifestyle with proper diet, exercise, and managing stress can significantly reduce the risk of developing heart disease.

    The **heart disease prediction model** uses various health indicators, such as age, cholesterol levels, blood pressure, and heart rate, to predict whether a person is likely to have heart disease or not. Early prediction can help in taking preventive measures to avoid heart-related issues.

    ### Key Risk Factors for Heart Disease:
    - High blood pressure
    - High cholesterol
    - Smoking
    - Lack of physical activity
    - Poor diet
    - Diabetes
    - Stress
                

     ![Heart Disease Image](https://www.ganeshdiagnostic.com/admin/assets/images/blog/Symptoms%20of%20Heart-related%20diseases_1677239775.jpg) 
                

    **Take care of your heart by maintaining a healthy lifestyle!**
                
    The following factors are considered for Diabetes prediction:
    """)

    # Empty default inputs
    age = st.number_input("Age", min_value=1, max_value=100, value=None, placeholder="Enter age")
    sex = st.selectbox("Sex", ["Select", "Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Select", 0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=None, placeholder="Enter BP")
    chol = st.number_input("Serum Cholesterol", min_value=100, max_value=500, value=None, placeholder="Enter cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Select", "Yes", "No"])
    restecg = st.selectbox("Resting ECG Results", ["Select", 0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=None, placeholder="Enter heart rate")
    exang = st.selectbox("Exercise Induced Angina", ["Select", "Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=None, placeholder="Enter oldpeak")
    slope = st.selectbox("Slope of ST Segment", ["Select", 0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", ["Select", 0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Select", 1, 2, 3])

    if st.button("Heart Disease Prediction Result"):
        # Check for empty fields
        if (
            None in [age, trestbps, chol, thalach, oldpeak]
            or "Select" in [sex, cp, fbs, restecg, exang, slope, ca, thal]
        ):
            st.warning("Please fill all fields before predicting.")
        else:
            # Convert categories
            sex_val = 1 if sex == "Male" else 0
            fbs_val = 1 if fbs == "Yes" else 0
            exang_val = 1 if exang == "Yes" else 0

            input_data = np.array([[
                age, sex_val, int(cp), trestbps, chol, fbs_val, int(restecg),
                thalach, exang_val, oldpeak, int(slope), int(ca), int(thal)
            ]])

            prediction = model.predict(input_data)

            if prediction[0] == 1:
                result = "The person is likely to have heart disease."
                st.error(f"Prediction: {result}")
            else:
                result = "The person is unlikely to have heart disease."
                st.success(f"Prediction: {result}")

            # Save to DB
            try:
                c.execute(''' 
                    INSERT INTO heart_disease_predictions (
                        age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal, result
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    age, sex_val, int(cp), trestbps, chol, fbs_val,
                    int(restecg), thalach, exang_val, oldpeak,
                    int(slope), int(ca), int(thal), result
                ))
                conn.commit()

            except ValueError as e:
                st.error(f"Error in saving data: {e}")




# ------------------ PARKINSON'S SECTION ------------------
elif selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction")

        # Description and image for Parkinson's disease
    st.markdown("""
        ### Parkinson's Disease
        Parkinson's disease is a progressive nervous system disorder that affects movement.
        It develops gradually, often starting with a small tremor in one hand. Other symptoms may include:
        - Slowness of movement
        - Stiffness
        - Impaired balance and coordination

         ![Parkinson's Disease Image](https://img1.wsimg.com/isteam/ip/2d223910-d5a2-4589-92ce-4c337e4cdfd9/How%20to%20use%20Chatgpt%20(26).jpg/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)
        The following voice measurements are used to detect Parkinson's Disease:
    """)

    parkinson_input_labels = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR",
        "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    inputs = []
    input_valid = True

    # Collect inputs for all 22 features
    for label in parkinson_input_labels:
        value = st.text_input(label)
        try:
            inputs.append(float(value))
        except:
            input_valid = False

    if st.button("Parkinson's Prediction Result"):
        if input_valid and len(inputs) == 22:
            input_np = np.array(inputs).reshape(1, -1)
            scaled_data = parkinsons_scaler.transform(input_np)
            prediction = parkinsons_model.predict(scaled_data)

            if prediction[0] == 1:
                result = "The person is likely to have Parkinson's disease"
            else:
                result = "The person is not likely to have Parkinson's disease"

            st.success(result)

            # Save to DB
            try:
                # Save inputs and result to DB
                c.execute('''
                    INSERT INTO parkinsons_predictions (input_data, result)
                    VALUES (?, ?)
                ''', (str(inputs), result))
                conn.commit()
            except Exception as e:
                st.error(f"An error occurred while saving to the database: {e}")

        else:
            st.error("Please fill all 22 inputs with valid numbers.")

# ------------------ HISTORY SECTION ------------------
elif selected == 'View History':
    st.title("📊 Previous Predictions History")

    # Diabetes History
    st.subheader("🩸 Diabetes Prediction History")
    c.execute("SELECT * FROM diabetes_predictions ORDER BY id DESC")
    rows = c.fetchall()
    if rows:
        for row in rows:
            st.write({
                'Pregnancies': row[1],
                'Glucose': row[2],
                'Blood Pressure': row[3],
                'Skin Thickness': row[4],
                'Insulin': row[5],
                'BMI': row[6],
                'DPF': row[7],
                'Age': row[8],
                'Result': row[9]
            })
    else:
        st.info("No diabetes predictions found.")

    st.markdown("---")

    # Parkinson's History
    st.subheader("🧠 Parkinson's Prediction History")
    c.execute("SELECT * FROM parkinsons_predictions ORDER BY id DESC")
    rows = c.fetchall()
    if rows:
        for row in rows:
            st.write({
                'Input Data': row[1],
                'Result': row[2]
            })
    else:
        st.info("No Parkinson's predictions found.")

    st.markdown("---")

    # Heart Disease History
    st.subheader("❤️ Heart Disease Prediction History")
    c.execute("SELECT * FROM heart_disease_predictions ORDER BY id DESC")
    rows = c.fetchall()
    if rows:
        for row in rows:
            st.write({
                'Age': row[1],
                'Sex': 'Male' if row[2] == 1 else 'Female',
                'Chest Pain Type': row[3],
                'Resting Blood Pressure': row[4],
                'Serum Cholesterol': row[5],
                'Fasting Blood Sugar > 120 mg/dl': 'Yes' if row[6] == 1 else 'No',
                'Resting ECG Results': row[7],
                'Max Heart Rate Achieved': row[8],
                'Exercise Induced Angina': 'Yes' if row[9] == 1 else 'No',
                'Oldpeak (ST depression)': row[10],
                'Slope of ST Segment': row[11],
                'Number of Major Vessels': row[12],
                'Thalassemia': row[13],
                'Result': row[14]
            })
    else:
        st.info("No heart disease predictions found.")





    