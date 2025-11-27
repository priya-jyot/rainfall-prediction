# ========================================================================
# ADVANCED STREAMLIT RAINFALL PREDICTION APP (FULL VISUALIZATION VERSION)
# ========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import warnings

warnings.filterwarnings("ignore")

# ========================================================================
# STREAMLIT CONFIG
# ========================================================================
st.set_page_config(page_title="Rainfall Prediction", layout="wide")
st.title("🌧️ Rainfall Prediction – Full ML Pipeline & Visualizations")

# ========================================================================
# UPLOAD CSV
# ========================================================================
uploaded_file = st.file_uploader("Upload weatherAUS.csv", type=["csv"])

if uploaded_file is None:
    st.info("Upload the dataset to begin...")
    st.stop()

# ========================================================================
# LOAD DATA
# ========================================================================
full_data = pd.read_csv(uploaded_file)

st.header("📌 Raw Data Overview")
st.dataframe(full_data.head())
st.write(f"Dataset Shape: **{full_data.shape}**")

# ========================================================================
# INITIAL PROCESSING
# ========================================================================
full_data['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
full_data['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

# ========================================================================
# CLASS DISTRIBUTION BEFORE OVERSAMPLING
# ========================================================================
st.subheader("🔍 Class Distribution Before Oversampling")
fig, ax = plt.subplots()
full_data['RainTomorrow'].value_counts().plot(kind='bar', color=['skyblue', 'navy'], ax=ax)
plt.title("Before Oversampling")
st.pyplot(fig)

# ========================================================================
# OVERSAMPLING
# ========================================================================
no = full_data[full_data.RainTomorrow == 0]
yes = full_data[full_data.RainTomorrow == 1]
yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)
oversampled = pd.concat([no, yes_oversampled])

st.subheader("🔍 Class Distribution After Oversampling")
fig, ax = plt.subplots()
oversampled['RainTomorrow'].value_counts().plot(kind='bar', color=['skyblue', 'navy'], ax=ax)
plt.title("After Oversampling")
st.pyplot(fig)

# ========================================================================
# MISSING DATA HEATMAP
# ========================================================================
st.subheader("📊 Missing Data Heatmap")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(oversampled.isnull(), cmap="PuBu", cbar=False, ax=ax)
st.pyplot(fig)

# ========================================================================
# CATEGORICAL IMPUTATION
# ========================================================================
for col in oversampled.select_dtypes(include=['object']).columns:
    oversampled[col] = oversampled[col].fillna(oversampled[col].mode()[0])

# Label Encoding
for col in oversampled.select_dtypes(include=['object']).columns:
    oversampled[col] = LabelEncoder().fit_transform(oversampled[col])

# ========================================================================
# MICE IMPUTATION
# ========================================================================
mice_imputer = IterativeImputer()
MiceImputed = oversampled.copy()
MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversampled)

# ========================================================================
# OUTLIER REMOVAL USING IQR
# ========================================================================
Q1 = MiceImputed.quantile(0.25)
Q3 = MiceImputed.quantile(0.75)
IQR = Q3 - Q1
MiceImputed = MiceImputed[~((MiceImputed < (Q1 - 1.5 * IQR)) |
                            (MiceImputed > (Q3 + 1.5 * IQR))).any(axis=1)]

st.write(f"Shape after Outlier Removal: **{MiceImputed.shape}**")

# ========================================================================
# CORRELATION HEATMAP
# ========================================================================
st.subheader("📈 Correlation Matrix Heatmap")
corr = MiceImputed.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, square=True, ax=ax)
st.pyplot(fig)

# ========================================================================
# FEATURE SELECTION (CHI SQUARE + RANDOM FOREST)
# ========================================================================
st.subheader("⭐ Feature Selection Results")

X = MiceImputed.drop("RainTomorrow", axis=1)
y = MiceImputed["RainTomorrow"]

# Chi-square selection
scaled = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=10).fit(scaled, y)
chi_features = X.columns[chi_selector.get_support()].tolist()

st.write("🔹 **Chi-Square Selected Features:**")
st.code(chi_features)

# Random Forest Feature Selection
rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100)).fit(X, y)
rf_features = X.columns[rf_selector.get_support()].tolist()

st.write("🔹 **Random Forest Selected Features:**")
st.code(rf_features)

# ========================================================================
# TRAIN-TEST SPLIT
# ========================================================================
selected_features = rf_features  # Using RF-selected features
X = MiceImputed[selected_features]
y = MiceImputed["RainTomorrow"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ========================================================================
# MODEL TRAINING FUNCTION
# ========================================================================
def train_and_display(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    st.write(f"### 🔵 {name} Results")
    st.write(f"Accuracy: **{accuracy:.4f}**")
    st.write(f"ROC AUC: **{roc_auc:.4f}**")
    st.write(f"Cohen's Kappa: **{kappa:.4f}**")

    # Classification Report
    st.code(classification_report(y_test, y_pred))

    # Confusion Matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    st.pyplot(fig)

    # ROC Curve
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0, 1], [0, 1], linestyle="--")
    st.pyplot(fig)

    return accuracy, roc_auc, kappa


# ========================================================================
# TRAIN ALL MODELS
# ========================================================================
models = {
    "Logistic Regression": LogisticRegression(solver="liblinear"),
    "Decision Tree": DecisionTreeClassifier(max_depth=16),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "LightGBM": lgb.LGBMClassifier(),
    "CatBoost": cb.CatBoostClassifier(verbose=0),
    "XGBoost": xgb.XGBClassifier()
}

results = {}
st.header("🚀 Model Training & Evaluation")

for name, model in models.items():
    results[name] = train_and_display(model, name)

# ========================================================================
# MODEL COMPARISON PLOTS
# ========================================================================
st.header("📊 Model Performance Comparison")

df_results = pd.DataFrame(results, index=["Accuracy", "ROC_AUC", "Kappa"]).T
st.dataframe(df_results)

# Accuracy Bar Plot
fig, ax = plt.subplots()
sns.barplot(x=df_results.index, y=df_results["Accuracy"], ax=ax)
plt.xticks(rotation=45)
plt.title("Accuracy Comparison")
st.pyplot(fig)

# ROC AUC Plot
fig, ax = plt.subplots()
sns.barplot(x=df_results.index, y=df_results["ROC_AUC"], ax=ax)
plt.xticks(rotation=45)
plt.title("ROC AUC Comparison")
st.pyplot(fig)

# Kappa Plot
fig, ax = plt.subplots()
sns.barplot(x=df_results.index, y=df_results["Kappa"], ax=ax)
plt.xticks(rotation=45)
plt.title("Cohen's Kappa Comparison")
st.pyplot(fig)

