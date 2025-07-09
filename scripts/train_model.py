# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
# from imblearn.over_sampling import SMOTE
# import joblib
# import warnings
# warnings.filterwarnings("ignore")

# def load_and_preprocess_data():
#     """Load and preprocess the dataset"""
#     df = pd.read_csv('telco_churn.csv')
    
#     # Drop unnecessary columns
#     df.drop('customerID', axis=1, inplace=True)
    
#     # Convert TotalCharges to numeric and handle missing values
#     df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
#     df.dropna(inplace=True)
    
#     # Encode target variable
#     df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
#     # Encode binary columns
#     binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
#     label_encoders = {}
#     for col in binary_cols:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#         label_encoders[col] = le
    
#     # One-hot encode remaining categorical variables
#     df_encoded = pd.get_dummies(df, drop_first=True)
    
#     return df, df_encoded, label_encoders

# def train_models():
#     """Train multiple models and return the best one"""
#     df, df_encoded, label_encoders = load_and_preprocess_data()
    
#     # Features and target
#     X = df_encoded.drop('Churn', axis=1)
#     y = df_encoded['Churn']
    
#     # Feature scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Handle class imbalance with SMOTE
#     smote = SMOTE(random_state=42)
#     X_res, y_res = smote.fit_resample(X_scaled, y)
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
#     # Train Random Forest (your original model)
#     param_grid_rf = {
#         'n_estimators': [100, 200],
#         'max_depth': [10, 15],
#         'min_samples_split': [2, 5],
#         'min_samples_leaf': [1, 3],
#         'max_features': ['sqrt']
#     }
    
#     rf = RandomForestClassifier(random_state=42)
#     grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
#     grid_rf.fit(X_train, y_train)
    
#     best_rf = grid_rf.best_estimator_
#     rf_pred = best_rf.predict(X_test)
#     rf_accuracy = accuracy_score(y_test, rf_pred)
#     rf_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
    
#     print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
#     print(f"Random Forest AUC: {rf_auc:.4f}")
    
#     # Save the model and preprocessing objects
#     joblib.dump(best_rf, 'best_model.pkl')
#     joblib.dump(scaler, 'scaler.pkl')
#     joblib.dump(label_encoders, 'label_encoders.pkl')
#     joblib.dump(X.columns.tolist(), 'feature_names.pkl')
    
#     # Save test data for evaluation
#     joblib.dump((X_test, y_test), 'test_data.pkl')
    
#     return {
#         'model': best_rf,
#         'scaler': scaler,
#         'label_encoders': label_encoders,
#         'feature_names': X.columns.tolist(),
#         'test_accuracy': rf_accuracy,
#         'test_auc': rf_auc,
#         'test_data': (X_test, y_test)
#     }

# if __name__ == "__main__":
#     results = train_models()
#     print("Model training completed successfully!")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv('telco_churn.csv')
    
    # Store original data for reference
    df_original = df.copy()
    
    # Drop unnecessary columns
    df.drop('customerID', axis=1, inplace=True)
    
    # Convert TotalCharges to numeric and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Identify and store column information for consistent preprocessing
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in binary_cols and col != 'Churn']
    
    print(f"Binary columns: {binary_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Encode binary columns
    label_encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {le.classes_}")
    
    # One-hot encode remaining categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Store preprocessing information
    preprocessing_info = {
        'binary_cols': binary_cols,
        'categorical_cols': categorical_cols,
        'feature_columns': [col for col in df_encoded.columns if col != 'Churn'],
        'original_columns': df_original.columns.tolist()
    }
    
    return df, df_encoded, label_encoders, preprocessing_info

def preprocess_single_customer(customer_data, label_encoders, preprocessing_info, scaler):
    """Preprocess a single customer's data for prediction"""
    
    # Convert to DataFrame if it's a dictionary
    if isinstance(customer_data, dict):
        df_customer = pd.DataFrame([customer_data])
    else:
        df_customer = customer_data.copy()
    
    # Ensure all expected columns are present
    expected_cols = preprocessing_info['original_columns']
    for col in expected_cols:
        if col not in df_customer.columns and col != 'customerID':
            # Add missing columns with default values
            if col in ['SeniorCitizen']:
                df_customer[col] = 0
            else:
                df_customer[col] = 'No'  # Default for most categorical
    
    # Convert TotalCharges to numeric
    if 'TotalCharges' in df_customer.columns:
        df_customer['TotalCharges'] = pd.to_numeric(df_customer['TotalCharges'], errors='coerce')
        df_customer['TotalCharges'].fillna(df_customer['TotalCharges'].median(), inplace=True)
    
    # Encode binary columns using the same encoders
    for col in preprocessing_info['binary_cols']:
        if col in df_customer.columns:
            le = label_encoders[col]
            # Handle unseen categories by using the first class
            df_customer[col] = df_customer[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
            )
    
    # One-hot encode categorical columns
    if preprocessing_info['categorical_cols']:
        df_customer = pd.get_dummies(df_customer, columns=preprocessing_info['categorical_cols'], drop_first=True)
    
    # Ensure all feature columns are present in the correct order
    for col in preprocessing_info['feature_columns']:
        if col not in df_customer.columns:
            df_customer[col] = 0
    
    # Select only the feature columns in the correct order
    df_customer = df_customer[preprocessing_info['feature_columns']]
    
    # Scale the features
    df_customer_scaled = scaler.transform(df_customer)
    
    return df_customer_scaled

def train_models():
    """Train multiple models and return the best one"""
    df, df_encoded, label_encoders, preprocessing_info = load_and_preprocess_data()
    
    # Features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    print(f"Training data shape: {X.shape}")
    print(f"Feature columns: {X.columns.tolist()}")
    print(f"Churn rate: {y.mean():.2%}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    print(f"After SMOTE - Shape: {X_res.shape}, Churn rate: {y_res.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    
    # Train Random Forest with better parameters
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_rf.fit(X_train, y_train)
    
    best_rf = grid_rf.best_estimator_
    rf_pred = best_rf.predict(X_test)
    rf_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    
    print(f"Best parameters: {grid_rf.best_params_}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Test with sample customers to verify different predictions
    print("\n" + "="*50)
    print("TESTING SAMPLE PREDICTIONS")
    print("="*50)
    
    test_customers = [
        {
            'name': 'High Risk Customer',
            'data': {
                'gender': 'Female',
                'SeniorCitizen': 0,
                'Partner': 'No',
                'Dependents': 'No',
                'tenure': 2,
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'Fiber optic',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'No',
                'DeviceProtection': 'No',
                'TechSupport': 'No',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 85.0,
                'TotalCharges': 170.0
            }
        },
        {
            'name': 'Low Risk Customer',
            'data': {
                'gender': 'Male',
                'SeniorCitizen': 1,
                'Partner': 'Yes',
                'Dependents': 'Yes',
                'tenure': 60,
                'PhoneService': 'Yes',
                'MultipleLines': 'Yes',
                'InternetService': 'DSL',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'Yes',
                'TechSupport': 'Yes',
                'StreamingTV': 'Yes',
                'StreamingMovies': 'Yes',
                'Contract': 'Two year',
                'PaperlessBilling': 'No',
                'PaymentMethod': 'Bank transfer (automatic)',
                'MonthlyCharges': 45.0,
                'TotalCharges': 2700.0
            }
        }
    ]
    
    for customer in test_customers:
        print(f"\n{customer['name']}:")
        try:
            customer_processed = preprocess_single_customer(
                customer['data'], label_encoders, preprocessing_info, scaler
            )
            churn_prob = best_rf.predict_proba(customer_processed)[0, 1]
            print(f"  Churn Probability: {churn_prob:.1%}")
            print(f"  Contract: {customer['data']['Contract']}")
            print(f"  Tenure: {customer['data']['tenure']} months")
            print(f"  Monthly Charges: ${customer['data']['MonthlyCharges']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save the model and preprocessing objects
    joblib.dump(best_rf, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(preprocessing_info, 'preprocessing_info.pkl')  # Save preprocessing info
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')
    
    # Save test data for evaluation
    joblib.dump((X_test, y_test), 'test_data.pkl')
    
    return {
        'model': best_rf,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'preprocessing_info': preprocessing_info,
        'feature_names': X.columns.tolist(),
        'test_accuracy': rf_accuracy,
        'test_auc': rf_auc,
        'test_data': (X_test, y_test)
    }

if __name__ == "__main__":
    results = train_models()
    print("\nModel training completed successfully!")
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Final Test AUC: {results['test_auc']:.4f}")
