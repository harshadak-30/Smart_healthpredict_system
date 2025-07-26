import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import os

# ----------------------------
# Train Diabetes Model
# ----------------------------
diabetes_df = pd.read_csv('dataset/diabetes.csv')
X_diabetes = diabetes_df.drop('Outcome', axis=1)
y_diabetes = diabetes_df['Outcome']

X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
diabetes_model = RandomForestClassifier()
diabetes_model.fit(X_train_dia, y_train_dia)

os.makedirs('models', exist_ok=True)
joblib.dump(diabetes_model, 'models/diabetes_model.pkl')
print("Diabetes model trained and saved.")

# ----------------------------
# Train Heart Disease Model
# ----------------------------
heart_df = pd.read_csv('dataset/heart.csv')
X_heart = heart_df.drop('target', axis=1)
y_heart = heart_df['target']

X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
heart_model = RandomForestClassifier()
heart_model.fit(X_train_heart, y_train_heart)

joblib.dump(heart_model, 'models/heart_model.pkl')
print("Heart disease model trained and saved.")

# ----------------------------
# Train Symptom Chatbot Model
# ----------------------------
df = pd.read_csv('dataset/symptom_disease.csv')
print(f"Columns in symptom CSV: {df.columns}")

df.rename(columns={'Symptom': 'symptom', 'weight': 'label'}, inplace=True)

vectorizer = CountVectorizer()
X_symptoms = vectorizer.fit_transform(df['symptom'])
y_disease = df['label']

chatbot_model = MultinomialNB()
chatbot_model.fit(X_symptoms, y_disease)

joblib.dump(chatbot_model, 'models/chatbot_model.pkl')
joblib.dump(vectorizer, 'models/symptom_vectorizer.pkl')
print("Chatbot model and vectorizer trained and saved.")

#-----liver dataset----
# Load dataset
df = pd.read_csv('dataset/liver.csv')

# Clean and preprocess
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df = df.dropna()

X = df.drop(['Dataset'], axis=1)
y = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)  # 1 = Liver disease, 0 = Normal

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/liver_model.pkl')
print("Liver model saved successfully.")

#----breast cancer---
# Breast Cancer Prediction Training

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
cancer_data = pd.read_csv("dataset/breast.csv")

# Optional: Drop ID column if it exists
if 'id' in cancer_data.columns:
    cancer_data.drop('id', axis=1, inplace=True)

# Select only the 10 main features
main_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean',
    'area_mean', 'concavity_mean', 'concave points_mean',
    'radius_worst', 'perimeter_worst', 'area_worst', 'concave points_worst'
]
X = cancer_data[main_features]
y = cancer_data['diagnosis'].replace({'M': 1, 'B': 0})  # Malignant = 1, Benign = 0

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
cancer_model = RandomForestClassifier()
cancer_model.fit(X_train, y_train)

# Save model
with open('models/breast_model.pkl', 'wb') as f:
    joblib.dump(cancer_model, f)

print("Breast Cancer Model Trained & Saved.")




#----stroke---
# Stroke Prediction Model

# Load stroke dataset
stroke_data = pd.read_csv("dataset/stroke.csv")
if 'id' in stroke_data.columns:
    stroke_data = stroke_data.drop('id', axis=1)
# Handle categorical variables (encode strings to numbers)
stroke_data.replace({'gender': {'Male': 0, 'Female': 1, 'Other': 2},
                     'ever_married': {'No': 0, 'Yes': 1},
                     'work_type': {'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3, 'Self-employed': 4},
                     'Residence_type': {'Urban': 0, 'Rural': 1},
                     'smoking_status': {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}}, inplace=True)

# Drop missing values (or impute them)
stroke_data.dropna(inplace=True)

# Features and target
X_stroke = stroke_data.drop('stroke', axis=1)
y_stroke = stroke_data['stroke']

# Split dataset
X_train_stroke, X_test_stroke, y_train_stroke, y_test_stroke = train_test_split(
    X_stroke, y_stroke, test_size=0.2, random_state=42)

# Train model
stroke_model = RandomForestClassifier()
stroke_model.fit(X_train_stroke, y_train_stroke)

# Save model
with open('models/stroke_model.pkl', 'wb') as f:
    joblib.dump(stroke_model, f)

print("Stroke prediction model trained and saved as stroke_model.pkl")
