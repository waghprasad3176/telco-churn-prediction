import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data.csv'
TARGET = 'Churn'

def load_data():
    df = pd.read_csv(DATA)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.drop(columns=['customerID'], errors='ignore')
    df[TARGET] = df[TARGET].map({'No': 0, 'Yes': 1})
    return df

def main():
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    numeric = X.select_dtypes(include=['number']).columns.tolist()
    categorical = X.select_dtypes(exclude=['number']).columns.tolist()
    numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('num', numeric_pipe, numeric), ('cat', categorical_pipe, categorical)])
    model = Pipeline([('preprocessor', prep), ('classifier', RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'precision': round(float(precision_score(y_test, pred)), 4),
        'recall': round(float(recall_score(y_test, pred)), 4),
        'f1_score': round(float(f1_score(y_test, pred)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, proba)), 4),
        'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
        'classification_report': classification_report(y_test, pred, target_names=['Retained', 'Churned'], output_dict=True),
        'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)), 'features': X.columns.tolist()
    }
    joblib.dump(model, BASE / 'churn_model.joblib')
    (BASE / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k != 'classification_report'}, indent=2))
    print('Saved churn_model.joblib and metrics.json')

if __name__ == '__main__': main()
