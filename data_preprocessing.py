import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(df.columns)
    print(df.head())

    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Convert numeric columns safely
    numeric_cols = [
        'no_of_trainings', 'age', 'previous_year_rating',
        'length_of_service', 'awards_won?', 'avg_training_score'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).astype(int)

    # Convert is_promoted ONLY if present
    if 'is_promoted' in df.columns:
        df['is_promoted'] = df['is_promoted'].astype(int)

    # Label encode categorical features
    cat_columns = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_columns:
        df[col] = le.fit_transform(df[col])

    return df
