
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from tensorflow.keras.models import load_model 

# ---------------- Load Saved Data & Models ----------------
st.title(" CIFAR-10 Binary Classifier Dashboard")

# Load test data
X_test = joblib.load("saved_models/X_test_raw.pkl")
y_test = joblib.load("saved_models/y_test.pkl")

X_test_knn = joblib.load("saved_models/X_test_knn.pkl")
X_test_log = joblib.load("saved_models/X_test_log.pkl")
X_test_cnn = joblib.load("saved_models/X_test_cnn.pkl")

# Load models
best_knn = joblib.load("saved_models/best_knn_model.joblib")
best_log = joblib.load("saved_models/best_log_model.joblib")

# Proper CNN loading
@st.cache_resource
def load_cnn():
    return load_model("saved_models/best_cnn_model.h5")

best_cnn = load_cnn()

# Load metrics
knn_metrics = joblib.load("saved_models/knn_metrics.pkl")
log_metrics = joblib.load("saved_models/log_metrics.pkl")
cnn_metrics = joblib.load("saved_models/cnn_metrics.pkl")

# Load preprocessing tools
scaler = joblib.load("saved_models/scaler.pkl")
pca = joblib.load("saved_models/pca.pkl")

# ---------------- Prediction Section ----------------
st.header(" Predict Image Class with Saved Models")
st.header("Select the number between 0-9999")

index = st.number_input("Select test image index", 0, len(X_test) - 1, 0)

if st.button("Predict"):
    img = X_test[index]
    true_label = 'Animal' if y_test[index] == 1 else 'Vehicle'

    st.image(img, caption=f"Actual Label: {true_label}", width=300)

    # Preprocessing for predictions
    img_flat = img.reshape(1, -1).astype('float32') / 255.0
    img_scaled = scaler.transform(img_flat)
    img_pca = pca.transform(img_scaled)
    img_cnn = img.astype('float32') / 255.0

    # Make predictions
    knn_model = best_knn['model']
    knn_pred = knn_model.predict(img_pca)[0]
    knn_prob = knn_model.predict_proba(img_pca)[0][1]

    log_pred = best_log.predict(img_pca)[0]
    log_prob = best_log.predict_proba(img_pca)[0][1]

    cnn_prob = best_cnn.predict(img_cnn[np.newaxis, ...])[0][0]
    cnn_pred = int(cnn_prob > 0.5)

    st.markdown("###  Model Predictions")
    st.write(f"KNN Prediction: **{'Animal' if knn_pred == 1 else 'Vehicle'}** (Confidence: {knn_prob:.2f})")
    st.write(f"Logistic Regression Prediction: **{'Animal' if log_pred == 1 else 'Vehicle'}** (Confidence: {log_prob:.2f})")
    st.write(f"CNN Prediction: **{'Animal' if cnn_pred == 1 else 'Vehicle'}** (Confidence: {cnn_prob:.2f})")

# ---------------- Evaluation Metrics Section ----------------
st.header(" Evaluation Metrics Overview")

def plot_roc_pr_curves(metrics, model_name):
    fpr, tpr = metrics['roc_curve']
    precision, recall = metrics['pr_curve']
    auc_roc = metrics['roc_auc']
    auc_pr = metrics['pr_auc']

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    axs[0].plot(fpr, tpr, label=f"AUC = {auc_roc:.3f}")
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='grey')
    axs[0].set_title(f"{model_name} ROC Curve")
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend(loc="lower right")

    # PR Curve
    axs[1].plot(recall, precision, label=f"AUC = {auc_pr:.3f}")
    axs[1].set_title(f"{model_name} Precision-Recall Curve")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].legend(loc="lower left")

    st.pyplot(fig)

# Accuracy comparison
st.subheader("Accuracy Comparison")
accs = {
    "KNN": knn_metrics["accuracy"],
    "Logistic Regression": log_metrics["accuracy"],
    "CNN": cnn_metrics["accuracy"]
}
fig_acc, ax_acc = plt.subplots()
ax_acc.bar(accs.keys(), accs.values(), color=['blue', 'orange', 'green'])
ax_acc.set_ylim([0, 1])
ax_acc.set_ylabel("Accuracy")
st.pyplot(fig_acc)

# ROC and PR Curves
st.subheader(" ROC and Precision-Recall Curves")
plot_roc_pr_curves(knn_metrics, "KNN")
plot_roc_pr_curves(log_metrics, "Logistic Regression")
plot_roc_pr_curves(cnn_metrics, "CNN")

st.markdown("---")
st.markdown("Developed with  to demonstrate model performance and predictions on CIFAR-10 binary classification.")
