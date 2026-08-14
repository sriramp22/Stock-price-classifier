import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_path = os.path.join(BASE_DIR, 'data', 'processed', 'combined.csv')
feature_cols = ['return_1d', 'ma5', 'ma20', 'vol5', 'vol20', 'vol_zscore', 'rsi14']
target_col = 'target'

def load_data():
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    print(f'Loaded {len(df)} rows, {df.shape[1]} columns')
    return df

def split_data(df):
    train = df[df.index < '2022-01-01']
    val = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')]
    test = df[df.index >= '2023-01-01']
    
    X_train, y_train = train[feature_cols], train[target_col]
    X_val, y_val = val[feature_cols], val[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    
    print(f'Train: {len(X_train)} rows')
    print(f'Val: {len(X_val)} rows')
    print(f'Test: {len(X_test)} rows')
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def train_and_evaluate(model, model_name, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    metrics = {'model': model_name,
               'accuracy': accuracy_score(y_val, y_pred),
               'precision': precision_score(y_val, y_pred),
               'recall': recall_score(y_val, y_pred),
               'f1': f1_score(y_val, y_pred)}
    
    print(f'\n{model_name}')
    print(f'Accuracy: {metrics['accuracy']:.4f}')
    print(f'Precision: {metrics['precision']:.4f}')
    print(f'Recall: {metrics['recall']:.4f}')
    print(f'F1: {metrics['f1']:.4f}')
    
    return metrics

def main():
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)
    results = []
    
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    results.append(train_and_evaluate(logreg, 'Logistic Regression', X_train, y_train, X_val, y_val))
    
    rand_forest = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)
    results.append(train_and_evaluate(rand_forest, "Random Forest", X_train, y_train, X_val, y_val))
    
    print('\n' + '=' * 50)
    print('Baseline Results Comparision')
    print('=' * 50)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
if __name__ == '__main__':
    main()