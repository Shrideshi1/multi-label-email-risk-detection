

##Imports
import os
import re
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
import tensorflow as tf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


##Data Imports


PROJECT_DIR = "/content/drive/.shortcut-targets-by-id/1SLhiH7VulyiPZ7L846kJBEbN1pa4c4zU/Multi_Label_Email_Risk_Detection"
FEATURE_DIR = f"{PROJECT_DIR}/data/processed/features"
MODELS_DIR = f"{PROJECT_DIR}/models"
REPORTS_DIR = f"{PROJECT_DIR}/reports"
DISTILBERT_DIR = f"{MODELS_DIR}/distilbert_email_risk"


# Ensure reporting directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)


##Categories from Data Cleaning & Feature Engineering
risk_names = {
    "financial_risk": "Financial Information",
    "credential_risk": "Credentials or Authentication",
    "customer_info_risk": "Customer or Employee Information",
    "proprietary_risk": "Classified or Proprietary Information",
    "legal_risk": "Legal or Contract Information",
    "attachment_risk": "Sensitive Attachment",
    "phishing_spam_risk": "Phishing or Spam",
}

##Define target multi-label risk categories
risk_cols = [
    "financial_risk",
    "credential_risk",
    "customer_info_risk",
    "proprietary_risk",
    "legal_risk",
    "attachment_risk",
    "phishing_spam_risk",
]


##Define structural metadata
metadata_cols = [
    "text_length",
    "word_count",
    "num_digits",
    "num_dollar_signs",
    "num_uppercase",
    "num_exclamation",
    "num_question",
]


##Define rule-based booleans
rule_cols = [
    "has_attachment_ext",
    "has_money",
    "has_credential_terms",
    "has_customer_terms",
    "has_legal_terms",
    "has_internal_terms",
]


## HELPER FUNCTIONS
## ----------------


@st.cache_resource
def load_assets():
    """
    Load and cache core machine learning models, vectorizers, and tokenizers
    from disk to Streamlit application


    Returns:
        tuple: (vectorizer, scaler, models_dict, tokenizer)
    """
    vectorizer = joblib.load(f"{FEATURE_DIR}/tfidf_vectorizer.pkl")
    scaler = joblib.load(f"{FEATURE_DIR}/metadata_scaler.pkl")


    models = {
        "Logistic Regression": joblib.load(f"{MODELS_DIR}/logistic_regression_model.pkl"),
        "Random Forest": joblib.load(f"{MODELS_DIR}/random_forest_model.pkl"),
        "SVM": joblib.load(f"{MODELS_DIR}/svm_model.pkl"),
        "XGBoost": joblib.load(f"{MODELS_DIR}/xgboost_model.pkl"),
        "Neural Network": tf.keras.models.load_model(f"{MODELS_DIR}/multi_label_email_risk_mlp.keras"),
        "DistilBERT": AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR),
    }
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    return vectorizer, scaler, models, tokenizer


##Asset loading
vectorizer, scaler, models, tokenizer = load_assets()


def clean_text(text):
    """
    Normalize raw email text by matching note 1 in Data Cleaning


    Args:
        text (str): Raw input email string.


    Returns:
        str: Cleaned and normalized text string.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[_\-=\*]{3,}", " ", text)
    text = re.sub(r"\b_+\w+_*\b", " ", text)
    text = re.sub(r"\b\w+_+\w+\b", " ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\-$]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_metadata_features_from_text(text):
    """
    Extract structural and statistical metadata from raw email


    Args:
        text (str): Raw input email string.


    Returns:
        pd.DataFrame: A single-row DataFrame containing extracted metadata features.
    """
    text = str(text)
    return pd.DataFrame([{
        "text_length": len(text),
        "word_count": len(text.split()),
        "num_digits": len(re.findall(r"\d", text)),
        "num_dollar_signs": text.count("$"),
        "num_uppercase": len(re.findall(r"[A-Z]", text)),
        "num_exclamation": text.count("!"),
        "num_question": text.count("?"),
    }])


def create_rule_features_from_text(text, has_attachments=False):
    """
    Extract rule-based binary indicators checking for specific keyword categories


    Args:
        text (str): Raw input email string.
        has_attachments (bool): Manual flag indicating if the email contains file attachments.


    Returns:
        pd.DataFrame: A single-row DataFrame containing binary rule flags.
    """
    text_lower = str(text).lower()
    has_attachment_ext = bool(re.search(r"\.pdf|\.docx|\.xlsx|\.csv|\.zip|\.pptx", text_lower))
    if has_attachments:
        has_attachment_ext = True


    return pd.DataFrame([{
        "has_attachment_ext": int(has_attachment_ext),
        "has_money": int(bool(re.search(r"\$| revenue | budget | valuation | invoice | payment ", text_lower))),
        "has_credential_terms": int(bool(re.search(r"password|token|api key|secret|credential|oauth|vpn|encryption key|root", text_lower))),
        "has_customer_terms": int(bool(re.search(r"customer|client|account|employee|vendor|partner|payroll", text_lower))),
        "has_legal_terms": int(bool(re.search(r"contract|nda|liability|clause|agreement|compliance|legal|indemnification", text_lower))),
        "has_internal_terms": int(bool(re.search(r"internal|confidential|do not forward|do not distribute|not for circulation", text_lower))),
    }])


def build_features(email_text, has_attachments=False):
    """
    Combine TF-IDF text vectors, scaled metadata features, and rule-based features
    into a single unified sparse feature matrix


    Args:
        email_text (str): Raw input email string.
        has_attachments (bool): Manual flag indicating attachment status.


    Returns:
        tuple: (cleaned_text, X_final_sparse_matrix, X_tfidf, metadata_df, rule_df)
    """
    cleaned = clean_text(email_text)
    X_tfidf = vectorizer.transform([cleaned])


    meta_df = create_metadata_features_from_text(email_text)
    meta_scaled = scaler.transform(meta_df[metadata_cols])
    meta_sparse = sp.csr_matrix(meta_scaled)


    rule_df = create_rule_features_from_text(email_text, has_attachments=has_attachments)
    rule_sparse = sp.csr_matrix(rule_df[rule_cols].values)


    X_final = sp.hstack([X_tfidf, meta_sparse, rule_sparse]).tocsr()
    return cleaned, X_final, X_tfidf, meta_df, rule_df


## MODEL PREDICTION FUNCTIONS
## --------------------------


def predict_sklearn_model(model, X_final):
    """
    Generate predictions and probabilities using Scikit-Learn models.


    Args:
        model: Trained scikit-learn estimator.
        X_final (scipy.sparse matrix): Combined feature matrix.


    Returns:
        tuple: (binary_predictions_array, probabilities_array_or_none)
    """
    preds = model.predict(X_final)[0]
    probas = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_final)
        except Exception:
            probas = None
    return preds, probas


def predict_keras_model(model, X_final):
    """
    Generate multi-label risk predictions using Neural Network model.


    Args:
        model: Trained model.
        X_final (scipy.sparse matrix): Combined feature matrix.


    Returns:
        tuple: (binary_predictions_array, probability_scores_array)
    """
    dense = X_final.toarray()
    probs = model.predict(dense, verbose=0)[0]
    preds = (probs > 0.5).astype(int)
    return preds, probs


def predict_distilbert_model(model, text):
    """
    Tokenize input text and generate multi-label risk predictions using
    the DistilBERT model.


    Args:
        model: Hugging Face sequence classification model.
        text (str): Cleaned or raw input email string.


    Returns:
        tuple: (binary_predictions_array, probability_scores_array)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    return preds, probs


def get_multioutput_probabilities(model, X_final):
    """
    Get probabilities for multi-output models of class 1 for each estimator


    Args:
        model: Multi-output sklearn estimator
        X_final: Feature Matrix
    Returns:
        np.array: Array of probabilities for each risk factor
    """
    return np.array([
        estimator.predict_proba(X_final)[0, list(estimator.classes_).index(1)]
        if 1 in estimator.classes_
        else float(estimator.classes_[0])
        for estimator in model.estimators_
    ])


##Feature Explanation and Reporting Utilities


def yes_no_series(s):
    """
    Map binary integer columns/series (1/0) to human-readable (Yes/No).


    Args:
        s (pd.Series): Pandas series containing binary values.


    Returns:
        pd.Series: Mapped series containing 'Yes' or 'No'.
    """
    return s.map({1: "Yes", 0: "No"})


def feature_names_all():
    """
    Combine TF-IDF vocabulary names with metadata and rule feature column names
    into an array for full feature space.


    Returns:
        np.array: Array of all feature names.
    """
    return np.array(list(vectorizer.get_feature_names_out()) + metadata_cols + rule_cols)


def top_feature_for_linear_model(model, X_final, preds, probas=None):
    """
    Identify the most influential feature for linear model risk prediction


    Args:
        model: Trained linear model with attribute coefficients (`coef_`).
        X_final (scipy.sparse matrix): Combined feature matrix.
        preds (np.array): Model prediction vector.
        probas (np.array, optional): Model probability scores.


    Returns:
        tuple: (risk_category_name, top_feature_name, contribution_score) or (None, None, None)
    """
    if not hasattr(model, "coef_"):
        return None, None, None
    pos_idx = np.where(preds == 1)[0]
    if len(pos_idx) == 0:
        return None, None, None


    if probas is not None:
        if isinstance(probas, np.ndarray) and probas.ndim == 2 and probas.shape[1] == 2:
            scores = probas[:, 1]
        elif isinstance(probas, np.ndarray) and probas.ndim == 1:
            scores = probas
        else:
            scores = preds.astype(float)
    else:
        scores = preds.astype(float)


    best_idx = pos_idx[np.argmax(scores[pos_idx])]
    coef_vec = model.coef_[best_idx]
    x_vec = X_final.toarray().ravel()
    contrib = x_vec * coef_vec
    fn = feature_names_all()
    top_i = int(np.argmax(contrib))
    return risk_cols[best_idx], fn[top_i], float(contrib[top_i])


def top_feature_for_tree_model(model):
    """
    Identify the most influential feature for tree based model risk prediction


    Args:
        model: Trained tree-based model with attribute `feature_importances_`.


    Returns:
        str or None: Name of the top feature, or None if unavailable.
    """
    if not hasattr(model, "feature_importances_"):
        return None
    fn = feature_names_all()
    top_i = int(np.argmax(model.feature_importances_))
    return fn[top_i]


## STREAMLIT USER INTERFACE SETUP
## ------------------------------
st.set_page_config(page_title="Email Risk Analyzer", layout="centered")
st.title("Email Risk Analyzer")
st.write("Paste an email, choose whether it has attachments, and run all models.")


##Input fields for raw email and manual attachment flag
email_text = st.text_area("Email text", height=220, placeholder="Paste the full email here...")
has_attachments = st.checkbox("Email has attachments", value=False)


## MODEL PREDICTIONS AND RESULTS DISPLAY
## -------------------------------------


if st.button("Analyze Email"):
    if not email_text.strip():
        st.error("Please enter email text.")
    else:
        ##Features for models and clean text
        cleaned, X_final, X_tfidf, meta_df, rule_df = build_features(email_text, has_attachments)


        ##Display structural feature tables
        st.subheader("Feature summary")
        st.write("Combined feature shape:", X_final.shape)
        st.write("TF-IDF shape:", X_tfidf.shape)
        st.dataframe(meta_df)
        st.dataframe(rule_df)


        ##Get all probabilities for models
        model_probs = {}


        ##Iterate through all loaded models to evaluate the input email
        for name, model in models.items():
            ##DistilBERT
            if name == "DistilBERT":
                preds, probs = predict_distilbert_model(model, email_text)
                model_probs[name] = probs
                continue


            ##Neural Network
            if name == "Neural Network":
                preds, probs = predict_keras_model(model, X_final)
                model_probs[name] = probs
                continue


            ##Scikit-Learn Models (Logistic Regression, RF, SVM, XGBoost)
            preds, probas = predict_sklearn_model(model, X_final)

            ##Get Probabilities - FIXED VERSION
            if hasattr(model, "estimators_"):
                ##Get Probs for each label
                probs_array = []
                for estimator in model.estimators_:
                    if hasattr(estimator, "predict_proba") and 1 in estimator.classes_:
                        try:
                            prob = estimator.predict_proba(X_final)[0, 1]
                            probs_array.append(prob)
                        except Exception:
                            # Fallback: use prediction (0 or 1) not 0.5
                            pred = float(estimator.predict(X_final)[0])
                            probs_array.append(pred)
                    else:
                        # No probability method, use prediction
                        pred = float(estimator.predict(X_final)[0])
                        probs_array.append(pred)
                probs = np.array(probs_array)
                preds = (probs > 0.5).astype(int)
            elif hasattr(model, "predict_proba"):
                probas_raw = model.predict_proba(X_final)
                if probas_raw.ndim == 2 and probas_raw.shape[1] == 2:
                    probs = probas_raw[:, 1]
                else:
                    probs = probas_raw[0] if probas_raw.ndim == 1 else preds.astype(float)
            else:
                ##Just in Case use Preds
                probs = preds.astype(float)

            ##Store probabilities
            model_probs[name] = probs


        ##Ensemble Results
        st.subheader("Ensemble Results")


        if model_probs:
            ##Create probability dataframe
            probability_df = pd.DataFrame(model_probs, index=risk_cols)
            probability_df["Risk Category"] = [risk_names[r] for r in risk_cols]

            ##Calculate ensemble metrics
            probability_df["Ensemble Average"] = probability_df[list(model_probs.keys())].mean(axis=1)
            probability_df["Final Probability"] = probability_df[list(model_probs.keys())].max(axis=1)
            probability_df["Triggering Model"] = probability_df[list(model_probs.keys())].idxmax(axis=1)

            Threshold = 0.50
            probability_df["Above Threshold"] = probability_df["Final Probability"] >= Threshold
            probability_df["Risk Level"] = probability_df["Final Probability"].apply(
                lambda x: "🔴 High" if x >= 0.75 else ("🟡 Medium" if x >= 0.50 else "🟢 Low")
            )

            ##Create results display dataframe
            results_df = probability_df.reset_index().rename(columns={"index": "Risk Code"})
            results_df["Threshold"] = 0.50
            results_df["Prediction"] = np.where(results_df["Final Probability"] >= 0.50, "Detected", "Not Detected")
            results_df["Probability"] = results_df["Final Probability"].map("{:.2%}".format)
            results_df["Ensemble Average"] = results_df["Ensemble Average"].map("{:.2%}".format)
            results_df = results_df.sort_values("Final Probability", ascending=False)

            display_cols = ["Risk Category", "Probability", "Threshold", "Triggering Model", "Ensemble Average", "Prediction"]
            st.dataframe(results_df[display_cols], hide_index=True)


            ##Summary Statistics
            st.subheader("Summary")

            confidential_risks = ["financial_risk", "credential_risk", "customer_info_risk", "proprietary_risk", "legal_risk", "attachment_risk"]
            spam_probability = probability_df.loc["phishing_spam_risk", "Final Probability"]
            proprietary_probability = probability_df.loc["proprietary_risk", "Final Probability"]
            highest_leak_category = probability_df.loc[confidential_risks, "Final Probability"].idxmax()
            highest_leak_probability = probability_df.loc[highest_leak_category, "Final Probability"]
            triggering_model_for_leak = probability_df.loc[highest_leak_category, "Triggering Model"]

            detected_count = (probability_df["Final Probability"] >= Threshold).sum()

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Risks Above Threshold",
                    value=f"{detected_count} / {len(risk_cols)}",
                    delta=f"Threshold: {Threshold:.0%}"
                )
                st.write(f"*Number of risk categories exceeding {Threshold:.0%} probability*")

            with col2:
                highest_risk_name = risk_names[highest_leak_category]
                st.metric(
                    label="Highest Risk Category",
                    value=f"{highest_leak_probability:.1%}",
                    delta=highest_risk_name
                )
                st.write(f"*Detected by: {probability_df.loc[highest_leak_category, 'Triggering Model']}*")


            # Model Performance Ranking by Best Probability
            st.subheader("Model Ranking by Highest Confidence Prediction")

            # Calculate each model's best (highest) probability across all risks
            model_best_probs = {}
            model_best_risk = {}

            for model_name in model_probs.keys():
                best_idx = probability_df.loc[confidential_risks + ["phishing_spam_risk"], model_name].idxmax()
                best_prob = probability_df.loc[best_idx, model_name]
                model_best_probs[model_name] = best_prob
                model_best_risk[model_name] = risk_names[best_idx]

            # Sort models by their best probability (highest to lowest)
            sorted_models = sorted(model_best_probs.items(), key=lambda x: x[1], reverse=True)

            # Display ranking table
            ranking_data = []
            for rank, (model_name, best_prob) in enumerate(sorted_models, 1):
                best_risk_name = model_best_risk[model_name]
                exceeds_threshold = best_prob >= Threshold
                status = "✅ Detected" if exceeds_threshold else "❌ Below threshold"

                ranking_data.append({
                    "Rank": rank,
                    "Model": model_name,
                    "Best Probability": best_prob,
                    "Risk Category": best_risk_name,
                    "Status": status
                })

            ranking_df = pd.DataFrame(ranking_data)
            st.dataframe(
                ranking_df.style.format({"Best Probability": "{:.2%}"}),
                hide_index=True,
                use_container_width=True
            )

            # Visual ranking with progress bars
            st.write("**Model Confidence Visualization:**")

            for rank, (model_name, best_prob) in enumerate(sorted_models, 1):
                best_risk_name = model_best_risk[model_name]
                exceeds_threshold = best_prob >= Threshold

                col1, col2, col3 = st.columns([2, 3, 1])

                with col1:
                    st.write(f"**#{rank} {model_name}**")

                with col2:
                    st.progress(float(min(best_prob, 1.0)), text=f"Best: {best_risk_name}")

                with col3:
                    if exceeds_threshold:
                        st.success(f"✅ {best_prob:.1%}")
                    else:
                        st.info(f"❌ {best_prob:.1%}")


            st.metric(
                label="Phishing or Spam Probability",
                value=f"{spam_probability:.2%}",
            )
            st.write(f"Triggering model: {probability_df.loc['phishing_spam_risk', 'Triggering Model']}")

            st.metric(
                label="Classified or Proprietary Probability",
                value=f"{proprietary_probability:.2%}",
            )
            st.write(f"Triggering model: {probability_df.loc['proprietary_risk', 'Triggering Model']}")

            st.metric(
                label="Highest Information-Leak Risk",
                value=f"{risk_names[highest_leak_category]} ({highest_leak_probability:.2%})",
            )
            st.write(f"Triggering model: {triggering_model_for_leak}")


            ##Individual Model Predictions
            st.subheader("Individual Model Predictions")

            for name, model in models.items():
                st.markdown(f"### {name}")
                model_probs_current = model_probs[name]

                if name == "DistilBERT":
                    preds, probs = predict_distilbert_model(model, email_text)
                    df = pd.DataFrame({
                        "Risk": risk_cols,
                        "Predicted": yes_no_series(pd.Series(preds)).values,
                        "Probability": probs,
                    })
                    st.dataframe(df)

                    pos_idx = np.where(preds == 1)[0]
                    if len(pos_idx) > 0:
                        best_idx = pos_idx[np.argmax(probs[pos_idx])]
                        st.write(f"Highest risk: {risk_names[risk_cols[best_idx]]}")
                    else:
                        st.write("No positive risk labels predicted.")
                    st.info("Feature attribution for DistilBERT is not included.")


                elif name == "Neural Network":
                    preds, probs = predict_keras_model(model, X_final)
                    df = pd.DataFrame({
                        "Risk": risk_cols,
                        "Predicted": yes_no_series(pd.Series(preds)).values,
                        "Probability": probs,
                    })
                    st.dataframe(df)

                    pos_idx = np.where(preds == 1)[0]
                    if len(pos_idx) > 0:
                        best_idx = pos_idx[np.argmax(probs[pos_idx])]
                        st.write(f"Highest risk: {risk_names[risk_cols[best_idx]]}")
                    else:
                        st.write("No positive risk labels predicted.")
                    st.info("Feature attribution for the neural network is not included.")


                else:
                    ##Scikit-Learn Models - FIXED VERSION
                    preds, probas = predict_sklearn_model(model, X_final)

                    ##Get Probabilities
                    if hasattr(model, "estimators_"):
                        probs_array = []
                        for estimator in model.estimators_:
                            if hasattr(estimator, "predict_proba") and 1 in estimator.classes_:
                                try:
                                    prob = estimator.predict_proba(X_final)[0, 1]
                                    probs_array.append(prob)
                                except Exception:
                                    ##Use prediction (0 or 1) not 0.5
                                    pred = float(estimator.predict(X_final)[0])
                                    probs_array.append(pred)
                            else:
                                # No probability method, use prediction
                                pred = float(estimator.predict(X_final)[0])
                                probs_array.append(pred)
                        probs = np.array(probs_array)
                        preds = (probs > 0.5).astype(int)
                    elif hasattr(model, "predict_proba"):
                        probas_raw = model.predict_proba(X_final)
                        if probas_raw.ndim == 2 and probas_raw.shape[1] == 2:
                            probs = probas_raw[:, 1]
                        else:
                            probs = probas_raw[0] if probas_raw.ndim == 1 else preds.astype(float)
                        preds = (probs > 0.5).astype(int)
                    else:
                        preds = model.predict(X_final)[0]
                        probs = preds.astype(float)

                    df = pd.DataFrame({
                        "Risk": risk_cols,
                        "Risk Category": [risk_names[r] for r in risk_cols],
                        "Predicted": yes_no_series(pd.Series(preds)).values,
                        "Probability": probs,
                    })
                    st.dataframe(df)

                    pos_idx = np.where(preds == 1)[0]
                    if len(pos_idx) > 0:
                        if probas is not None and isinstance(probas, np.ndarray):
                            if probas.ndim == 2 and probas.shape[1] == 2:
                                scores = probas[:, 1]
                            elif probas.ndim == 1:
                                scores = probas
                            else:
                                scores = preds.astype(float)
                        else:
                            scores = preds.astype(float)

                        best_idx = pos_idx[np.argmax(scores[pos_idx])]
                        st.write(f"**Highest risk:** {risk_names[risk_cols[best_idx]]}")

                        ##Extract and display feature attributions based on model type
                        if hasattr(model, "coef_"):
                            risk_name, feat_name, contrib = top_feature_for_linear_model(model, X_final, preds, probas)
                            if feat_name is not None:
                                st.write(f"Top contributing feature for {risk_name}: {feat_name} ({contrib:.4f})")
                        elif hasattr(model, "feature_importances_"):
                            feat_name = top_feature_for_tree_model(model)
                            if feat_name is not None:
                                st.write(f"Top global feature: {feat_name}")
                    else:
                        st.write("No positive risk labels predicted.")
