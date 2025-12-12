# feature_selection.py

def select_features(df):
    # X = df.drop(['employee_id', 'is_promoted'], axis=1)
    # y = df['is_promoted']
    # Drop ID if available
    id_cols = ['employee_id', 'EmployeeNumber', 'emp_id']
    drop_ids = [col for col in id_cols if col in df.columns]

    # Drop target column only if it exists
    if 'is_promoted' in df.columns:
        X = df.drop(drop_ids + ['is_promoted'], axis=1)
        y = df['is_promoted']
    else:
        # Prediction mode (CSV without target column)
        X = df.drop(drop_ids, axis=1)
        y = None

    return X, y
