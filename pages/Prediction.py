import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

cb_labels = ['High', 'Low', 'Medium']
svm_labels = ['', 'Low', 'Medium', 'High']

#setup models
cb = CatBoostClassifier()
cb.load_model('models/catboost_tuned.cbm')
rf = joblib.load('models/rf_tuned.joblib')
sd_scl = joblib.load('models/scaler.joblib')
svm = joblib.load('models/base_ovr_73.joblib')
xg = XGBClassifier()
xg.load_model('models/xgb_baseline.json')
nb = joblib.load('models/gaussian_nb_best.joblib')


## PAGE
st.set_page_config(
    page_title='Model Predictions',
    layout='wide'
)
st.title('Prediction of Online Gaming Behaviour')
st.write('Predict the Engagement Level of players using CatBoost, RandomForest and SVM models.')
st.write('Upload a .csv file to run multiple predictions at once, or remove the file to run single predictions.')

sessions_per_week = st.number_input('Sessions played per week', min_value=0, max_value=50)
avg_session_duration_mins = st.number_input('Average session duration (mins)', min_value=0, max_value=600)
achievements_unlocked = st.number_input('Achievements unlocked', min_value=0, max_value=100)
player_level = st.number_input('Player level', min_value=0, max_value=100)

st.write('Alternatively, upload a file below.')
csv = st.file_uploader('Upload .csv', type='csv')

if st.button('Predict'):
    try:
        X = pd.read_csv(csv)
    except:
        X = pd.DataFrame(
            {
                'SessionsPerWeek':[sessions_per_week],
                'AvgSessionDurationMinutes':[avg_session_duration_mins],
                'AchievementsUnlocked':[achievements_unlocked],
                'PlayerLevel':[player_level]
            }
        )
    
    rf_pred = rf.predict(X)
    cb_pred = cb.predict(X)
    svm_pred = svm.predict(sd_scl.transform(X))
    xg_pred = xg.predict(X)
    nb_pred = nb.predict(X)

    cb_strung = []
    xg_strung = []
    svm_strung = []

    for i in range(len(cb_pred)):
        cb_strung.append(cb_labels[int(cb_pred[i])])
        xg_strung.append(cb_labels[int(xg_pred[i])])
        svm_strung.append(svm_labels[int(svm_pred[i])])

    X['CatBoost'] = cb_strung
    X['RandomForest'] = rf_pred
    X['SVM'] = svm_strung
    X['XGBoost'] = xg_strung
    X['GaussianNB'] = nb_pred

    st.write(X)