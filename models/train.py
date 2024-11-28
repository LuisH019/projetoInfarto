import sys 
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import lightgbm as lgb
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

from utils.graphics import readCsv

def train():
    df = readCsv()
    df['HeartDisease'] = df['HeartDisease'].replace(-1, 0)

    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    smote_enn = SMOTEENN(random_state=42)
    X_train, y_train = smote_enn.fit_resample(X_train, y_train)

    model_lgbm = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        class_weight='balanced',
        random_state=42
    )

    model_lgbm.fit(X_train, y_train)

    joblib.dump(model_lgbm, 'models/model_lgbm.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("Modelo treinado e salvo com sucesso.")

train()


