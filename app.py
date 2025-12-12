# app.py

from flask import Flask, request, jsonify, render_template,redirect,url_for,session
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


from data_preprocessing import load_and_clean_data
from feature_selection import select_features

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models
random_forest_model = joblib.load("models/random_forest_model.pkl")
gradient_boosting_model = joblib.load("models/gradient_boosting_model.pkl")
xgboost_model = joblib.load("models/xgboost_model.pkl")

# Ensure uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dummy login credentials
USERS = {"admin": "password123"}


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if USERS.get(username) == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'], selected_model=None)


@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    file = request.files['file']
    model_choice = request.form.get('model_choice')

    if not file or not model_choice:
        return "No file uploaded", 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read raw CSV for display
    df_raw = pd.read_csv(file_path)

    # Clean column names (IMPORTANT FIX)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    # Drop is_promoted if accidentally present
    if 'is_promoted' in df_raw.columns:
        df_raw = df_raw.drop(columns=['is_promoted'])

    # Process dataset (without target column)
    df_processed = load_and_clean_data(file_path)

    # Feature selection - get only X
    X, _ = select_features(df_processed)

    # Select model
    if model_choice == 'random_forest':
        model = random_forest_model
    elif model_choice == 'gradient_boosting':
        model = gradient_boosting_model
    elif model_choice == 'xgboost':
        model = xgboost_model
    else:
        return "Invalid model selected", 400

    # Make predictions
    predictions = model.predict(X)

    df_raw['prediction'] = predictions

    # Sort highest prediction first
    df_raw.sort_values(by='prediction', ascending=False, inplace=True)

    prediction_table = df_raw.to_html(
        classes="table table-bordered table-striped",
        index=False
    )

    return render_template(
        'dashboard.html',
        username=session['username'],
        table=prediction_table,
        selected_model=model_choice
    )


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)

