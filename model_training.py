from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import pandas as pd

# Train Random Forest Model
def train_random_forest(X, y):
    print("\nTraining Random Forest with SMOTE for class imbalance...")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("\nRandom Forest Report with SMOTE:")
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, "models/random_forest_model.pkl")
    print("✅ Saved random_forest model to models/random_forest_model.pkl")

# Train Gradient Boosting Model
def train_gradient_boosting(X, y):
    print("\nTraining Gradient Boosting with SMOTE for class imbalance...")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    # Train the model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("\nGradient Boosting Report with SMOTE:")
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, "models/gradient_boosting_model.pkl")
    print("✅ Saved gradient_boosting model to models/gradient_boosting_model.pkl")

# Train XGBoost Model
def train_xgboost(X, y):
    print("\nTraining XGBOOST with SMOTE for class imbalance...")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    # Train the model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_resampled, y_resampled)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("\nXGBOOST Report with SMOTE:")
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, "models/xgboost_model.pkl")
    print("✅ Saved xgboost model to models/xgboost_model.pkl")

# Train all models
def train_models(X, y):
    train_random_forest(X, y)
    train_gradient_boosting(X, y)
    train_xgboost(X, y)
