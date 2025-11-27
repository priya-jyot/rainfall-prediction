📘 Rainfall Prediction using Machine Learning & Streamlit

This project predicts whether it will rain tomorrow based on historical weather data.
It includes a fully interactive Streamlit web application, complete EDA visualizations,
data preprocessing pipeline, feature selection, and multiple machine learning models
(Logistic Regression, Decision Tree, Random Forest, LightGBM, XGBoost, CatBoost, Neural Network).

The app allows you to upload the dataset, explore insights, visualize patterns,
train models, and compare their performance — all in a clean dashboard interface.

📂 Dataset

This project uses the weatherAUS.csv dataset.

👉 You must use the dataset provided in the repository.
Do not download externally — the preprocessing steps are mapped to the dataset in this repo.

🚀 Features
✔ Interactive Visualizations

Class distribution (before & after oversampling)

Missing data heatmap

Correlation heatmap

Numerical & categorical distributions

Boxplots for outlier detection

Pairplots (sampled for performance)

✔ Data Preprocessing

Handling missing values

Oversampling to fix class imbalance

MICE imputation

Outlier removal using IQR

Label encoding

Scaling with StandardScaler

✔ Machine Learning Models

Logistic Regression

Decision Tree

Neural Network (MLPClassifier)

Random Forest

LightGBM

CatBoost

XGBoost

✔ Model Evaluation

Confusion Matrix

ROC Curve

Accuracy

ROC AUC

Cohen’s Kappa

Full classification reports

Model comparison charts

🛠️ Tech Stack

Python 3.9+

Streamlit

Scikit-learn

LightGBM / XGBoost / CatBoost


Clone this repository:

git clone https://github.com/<your-username>/rainfall-prediction.git
cd rainfall-prediction

Install dependencies:

pip install -r requirements.txt

▶️ Run the Streamlit App
streamlit run app.py

📁 Project Structure
📦 rainfall-prediction
│
├── app.py                 # Main Streamlit application
├── weatherAUS.csv         # Dataset (use this only)
├── requirements.txt       # Dependencies
├── README.md              # Project documentation


🧠 Model Results

The app trains multiple ML models and compares them using:

Accuracy

ROC AUC

Cohen’s Kappa

The results help identify the best-performing algorithm for rainfall prediction

Seaborn + Matplotlib + Plotly

NumPy / Pandas
