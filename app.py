from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from models import db, PredictionEntry

db.init_app(app)

with app.app_context():
    db.create_all()

# ---------- Load Symptom DataFrame Globally ----------
try:
    df = pd.read_csv('dataset/symptom_disease.csv')
except Exception as e:
    print(f"[ERROR] Could not load symptom_disease.csv: {e}")
    df = None

# ---------- Home Page ----------
@app.route('/')
def home():
    entries = PredictionEntry.query.order_by(PredictionEntry.timestamp.desc()).all()
    return render_template('index.html', entries=entries)

@app.route('/dashboard')
def dashboard():
    entries = PredictionEntry.query.order_by(PredictionEntry.timestamp.desc()).all()
    return render_template('dashboard.html', entries=entries)


# ---------- Diabetes Prediction ----------
DIABETES_COLS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        username = request.form.get('username', 'Anonymous')
        age = request.form.get('Age', 0)
        features = [float(request.form.get(col)) for col in DIABETES_COLS]
        model = joblib.load('models/diabetes_model.pkl')
        prediction = model.predict([features])[0]
        suggestion = ""
        if prediction == 1:
            suggestion = "Maintain a low-sugar diet, exercise regularly, avoid stress, and monitor glucose levels. Consult a doctor for further treatment."
        # Save to database
        entry = PredictionEntry(username=username, age=age, disease='Diabetes', prediction=str(prediction))
        db.session.add(entry)
        db.session.commit()
        return render_template('result.html', disease='Diabetes', prediction=prediction, suggestion=suggestion)
    return render_template('form.html', disease='Diabetes', columns=['username'] + DIABETES_COLS)

# ---------- Heart Disease Prediction ----------
HEART_COLS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        username = request.form.get('username', 'Anonymous')
        age = request.form.get('age', 0)
        features = [float(request.form.get(col)) for col in HEART_COLS]
        model = joblib.load('models/heart_model.pkl')
        prediction = model.predict([features])[0]
        suggestion = ""
        if prediction == 1:
            suggestion = "Follow a heart-healthy diet, reduce salt, quit smoking, and get regular checkups. Consult a cardiologist."
        # Save to database
        entry = PredictionEntry(username=username, age=age, disease='Heart Disease', prediction=str(prediction))
        db.session.add(entry)
        db.session.commit()
        return render_template('result.html', disease='Heart Disease', prediction=prediction, suggestion=suggestion)
    return render_template('form.html', disease='Heart Disease', columns=['username', 'age'] + HEART_COLS)

# ---------- Chatbot Symptom Checker ----------
# Route to render chatbot UI
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# Function to analyze symptoms
def get_symptom_weights(user_input):
    global df
    try:
        print('DEBUG: df is None:', df is None)
        if df is not None:
            print('DEBUG: df shape:', df.shape)
            print('DEBUG: df head:', df.head())
        if df is None:
            return "Symptom database not available. Please contact admin."

        print('DEBUG: user_input:', user_input)
        # Clean input: split by comma, strip spaces, convert to lowercase
        # Validate user input
        if not user_input.strip():
            return "Please enter symptoms first."
        input_symptoms = [sym.strip().lower() for sym in user_input.split(',') if sym.strip()]
        print('DEBUG: input_symptoms:', input_symptoms)
        if not input_symptoms:
            return "Please enter symptoms first."

        # Normalize column
        df['Symptom'] = df['Symptom'].astype(str).str.strip().str.lower()

        # Match user symptoms
        matched = df[df['Symptom'].isin(input_symptoms)]
        print('DEBUG: matched shape:', matched.shape)
        print('DEBUG: matched head:', matched.head())

        # If no match
        if matched.empty:
            return "Sorry, none of the symptoms were recognized. Please try again."

        # Prepare response
        response = "Here's a quick assessment based on your symptoms:\n\n"
        for _, row in matched.iterrows():
            response += f"- {row['Symptom'].title()}: Severity {row['weight']}\n"

        # Suggestions
        tips = [
            "Stay hydrated.",
            "Get adequate rest.",
            "Eat nutritious food.",
            "Consult a doctor if symptoms persist."
        ]
        response += "\nSuggestions:\n" + "\n".join(f"- {tip}" for tip in tips)

        return response
    except Exception as e:
        import traceback
        print("Error in get_symptom_weights:", traceback.format_exc())
        return "Server error while analyzing symptoms."

# ---------- Liver Disease Prediction ----------
LIVER_COLS = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
    'Total_Proteins', 'Albumin', 'Albumin_and_Globulin_Ratio'
]

@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'POST':
        try:
            username = request.form.get('username', 'Anonymous')
            age = request.form.get('Age', 0)
            data = []
            for col in LIVER_COLS:
                value = request.form.get(col)
                if col == 'Gender':
                    value = 1 if value.lower() == 'male' else 0
                data.append(float(value))
            model = joblib.load('models/liver_model.pkl')
            prediction = model.predict([data])[0]
            suggestion = ""
            if prediction == 1:
                suggestion = "Abnormal liver detected. Please consult a hepatologist. Avoid alcohol, take a liver-friendly diet, and get regular checkups."
            else:
                suggestion = "Your liver seems to be functioning normally. Maintain a healthy lifestyle."
            # Save to database
            entry = PredictionEntry(username=username, age=age, disease='Liver Disease', prediction=str(prediction))
            db.session.add(entry)
            db.session.commit()
            return render_template('result.html', disease='Liver Disease', prediction=prediction, suggestion=suggestion)
        except Exception as e:
            return f"Error during prediction: {e}"
    return render_template('form.html', disease='Liver Disease', columns=['username'] + LIVER_COLS)
 
#----stroke ----
STROKE_COLS = [
    'gender', 'age', 'hypertension', 'heart_disease',
    'ever_married', 'work_type', 'Residence_type',
    'avg_glucose_level', 'bmi', 'smoking_status'
]

@app.route('/stroke', methods=['GET', 'POST'])
def stroke():
    if request.method == 'POST':
        try:
            # Encode and prepare data
            form = request.form
            gender = 1 if form.get('gender').lower() == 'male' else 0
            age = float(form.get('age'))
            hypertension = int(form.get('hypertension'))
            heart_disease = int(form.get('heart_disease'))
            ever_married = 1 if form.get('ever_married').lower() == 'yes' else 0
            work_type_map = {'private': 0, 'self-employed': 1, 'govt_job': 2, 'children': 3, 'never_worked': 4}
            work_type = work_type_map.get(form.get('work_type').lower(), 0)
            residence_type = 0 if form.get('Residence_type').lower() == 'urban' else 1
            avg_glucose_level = float(form.get('avg_glucose_level'))
            bmi = float(form.get('bmi'))
            smoking_status_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'unknown': 3}
            smoking_status = smoking_status_map.get(form.get('smoking_status').lower(), 3)
            features = [gender, age_val, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]
            model = joblib.load('models/stroke_model.pkl')
            prediction = model.predict([features])[0]
            suggestion = ""
            if prediction == 1:
                suggestion = "High risk of stroke. Please consult a neurologist immediately. Control blood pressure, maintain a healthy diet, and exercise."
            else:
                suggestion = "Low risk of stroke. Keep up a healthy lifestyle and regular check-ups."
            # Save to database
            entry = PredictionEntry(username=username, age=age, disease='Stroke', prediction=str(prediction))
            db.session.add(entry)
            db.session.commit()
            return render_template('result.html', disease='Stroke Prediction', prediction=prediction, suggestion=suggestion)
        except Exception as e:
            return f"Error during prediction: {e}"
    return render_template('form.html', disease='Stroke Prediction', columns=['username', 'age'] + STROKE_COLS)

# ------breast cancer ----
CANCER_COLS = [
    'radius_mean', 'texture_mean', 'perimeter_mean',
    'area_mean', 'concavity_mean', 'concave points_mean',
    'radius_worst', 'perimeter_worst', 'area_worst', 'concave points_worst'
]


@app.route('/cancer', methods=['GET', 'POST'])
def cancer():
    if request.method == 'POST':
        try:
            username = request.form.get('username', 'Anonymous')
            age = request.form.get('radius_mean', 0)  # No direct age field, fallback to 0
            data = [float(request.form[col]) for col in CANCER_COLS]
            model = joblib.load('models/breast_model.pkl')
            prediction = model.predict([data])[0]
            suggestion = (
                "High risk of malignant tumor. Please consult an oncologist for further tests and guidance."
                if prediction == 1 else
                "Likely benign tumor. Continue regular checkups."
            )
            # Save to database
            entry = PredictionEntry(username=username, age=age, disease='Breast Cancer', prediction=str(prediction))
            db.session.add(entry)
            db.session.commit()
            return render_template('result.html', disease='Breast Cancer', prediction=prediction, suggestion=suggestion)
        except Exception as e:
            return f"Error during prediction: {e}"
    return render_template('form.html', disease='Breast Cancer', columns=['username'] + CANCER_COLS)


# Route to receive AJAX request from chatbot and respond
@app.route('/chatbot-analyze', methods=['POST', 'GET'])
def chatbot_analyze():
    try:
        # Try to get user input from JSON, then form, then args
        user_input = ''
        if request.is_json:
            data = request.get_json()
            user_input = data.get('message', '') if data else ''
        elif request.form:
            user_input = request.form.get('message', '')
        elif request.args:
            user_input = request.args.get('message', '')

        if not user_input or not user_input.strip():
            return jsonify({'reply': "Please enter symptoms first."})

        result = get_symptom_weights(user_input)
        return jsonify({'reply': result})
    except Exception as e:
        import traceback
        print("Error in chatbot_analyze:", traceback.format_exc())
        return jsonify({'reply': "Internal server error."})


# ---------- Run App ----------
<<<<<<< HEAD

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))  # default port for Render
    app.run(host='0.0.0.0', port=port)
=======
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))  # default port for Render
    app.run(host='0.0.0.0', port=port)
>>>>>>> 0a3a4caa212ac4985e9ba4c37c8c65df1d918dc1
