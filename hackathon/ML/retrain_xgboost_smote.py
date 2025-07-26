import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
from textblob import TextBlob
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# ========== STEP 1: Load & Validate Data ==========
def load_data(csv_path="training_dataset_enhanced.csv"):
    df = pd.read_csv(csv_path, encoding='utf-8', quotechar='"')
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["category"].str.lower() != "category"]
    df = df[df["category"] != "feedback"]
    print("\n📊 Class Distribution:")
    print(df["category"].value_counts())
    return df

# ========== STEP 2: Enhanced Feature Engineering ==========
def add_features(df):
    df['email_length'] = df['text'].apply(len)
    df['num_questions'] = df['text'].apply(lambda x: x.count('?'))
    df['num_exclamations'] = df['text'].apply(lambda x: x.count('!'))
    df['contains_table'] = df['text'].apply(lambda x: int('Description' in x and 'Qty' in x or ':' in x))
    df['has_po'] = df['text'].apply(lambda x: 1 if re.search(r'(PO[\s#:/\-]*\d+|purchase\s*order)', x, re.IGNORECASE) else 0)
    df['has_gst'] = df['text'].apply(lambda x: 1 if re.search(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b', x) else 0)
    df['has_model'] = df['text'].apply(lambda x: 1 if re.search(r'model[\s:]*[A-Z0-9\-]+', x, re.IGNORECASE) else 0)
    df['has_urgent'] = df['text'].apply(lambda x: 1 if re.search(r'\burgen(t|cy)|immediate(ly)?|asap\b', x, re.IGNORECASE) else 0)
    df['has_project'] = df['text'].apply(lambda x: 1 if re.search(r'\bproject\b.*\b(name|code|id|#)\b', x, re.IGNORECASE) else 0)
    df['has_specs'] = df['text'].apply(lambda x: 1 if re.search(r'\b(specification|technical\s*details?|requirements?)\b', x, re.IGNORECASE) else 0)
    df['has_legal'] = df['text'].apply(lambda x: 1 if re.search(r'\b(terms\s*&?\s*conditions|contract|agreement|warranty)\b', x, re.IGNORECASE) else 0)
    df['urgency_score'] = df['text'].apply(lambda x: len(re.findall(r'\burgen(t|cy)|immediate(ly)?|asap\b', x.lower())))
    df['legal_terms'] = df['text'].apply(lambda x: int(bool(re.search(r'\b(terms|contract|agreement|warranty|po)\b', x.lower()))))
    df['project_mention'] = df['text'].apply(lambda x: int(bool(re.search(r'\bproject\b.*\b(name|code|id|#)', x.lower()))))
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['has_negative_tone'] = df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity < -0.5 else 0)
    df['complaint_keywords'] = df['text'].apply(
    lambda x: len(re.findall(r'\b(issue|problem|error|complaint|refund|replace|damage|wrong|missing|failed)\b', x.lower()))
    )
    df['negative_sentiment'] = df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity < -0.3 else 0)
    df['complaint_triggers'] = df['text'].apply(
    lambda x: len(re.findall(r'\b(issue|problem|error|failed|wrong|missing|damage|complaint)\b', x.lower()))
    )
    strong_complaints = [
    "unacceptable", "angry", "disappointed", "fed up", "this is not done",
    "we are unhappy", "very poor", "your team failed", "this must be fixed"
    ]
    df["strong_complaint_hits"] = df["text"].apply(
    lambda x: sum(1 for phrase in strong_complaints if phrase in x.lower())
    )

    df['feedback_triggers'] = df['text'].apply(
    lambda x: len(re.findall(r'\b(thanks|appreciate|good|great|smooth|perfect|excellent)\b', x.lower()))
    )
    df['invoice_mismatch'] = df['text'].apply(
        lambda x: 1 if re.search(r'invoice.*(match|mismatch|difference|discrepancy)', x.lower()) else 0
    )
    df['installation_mention'] = df['text'].apply(
        lambda x: 1 if re.search(r'install.*(team|crew|staff|service)', x.lower()) else 0
    )
    # 🔹 Pricing keyword flag
    pricing_keywords = ["price", "cost", "rate", "quotation", "quote", "charges"]
    df["has_pricing_word"] = df["text"].str.lower().apply(lambda x: int(any(word in x for word in pricing_keywords)))

    # 🔹 Certification enquiry flag
    cert_keywords = ["certification", "certified", "isi", "ce", "ul", "approval", "license", "standard", "compliance"]
    df["has_certification_request"] = df["text"].str.lower().apply(lambda x: int(any(word in x for word in cert_keywords)))

    return df

# ========== STEP 3: Vectorization ==========
def vectorize_text(texts):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words='english',
        min_df=3,
        max_features=8000,
        sublinear_tf=True,
        analyzer='word',
        token_pattern=r'(?u)\b[a-z][a-z0-9_]{2,}\b',
        max_df=0.7,
    )

    X_tfidf = vectorizer.fit_transform(texts)
     
    assert X_tfidf.shape[0] == len(texts), (
        f"Vectorization failed: expected {len(texts)} rows, got {X_tfidf.shape[0]}. "
        f"Possible empty documents or preprocessing issues."
    )
    
    return X_tfidf, vectorizer

# ========== STEP 4: Train XGBoost Model with SMOTE ==========
def train_xgboost(X, y, label_encoder, df):
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    
    print("\n📊 Class Distribution After SMOTE:")
    print(Counter(y_resampled))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, 
        stratify=y_resampled, 
        test_size=0.2, 
        random_state=42
    )
    
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        scale_pos_weight={0:1.5, 1:1, 2:1, 3:1, 4:1},
        use_label_encoder=False
    )
    model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\n🧩 Confusion Matrix:")
    labels = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')
    print(pd.DataFrame(cm, index=labels, columns=labels).round(2))

    print("\n🔍 Error Analysis (Top 5 Misclassified Samples):")
    misclassified = np.where(y_pred != y_test)[0]
    for i in misclassified[:5]:  # Print first 5 errors
        print(f"\nTrue: {label_encoder.inverse_transform([y_test[i]])[0]}")
        print(f"Predicted: {label_encoder.inverse_transform([y_pred[i]])[0]}")
        print(f"Text: {df.iloc[i]['text'][:200]}...")
    
    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    print("\n📊 Class-Wise Metrics:")
    prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name:20} Precision: {prec[i]:.2f} Recall: {recall[i]:.2f} F1: {f1[i]:.2f}")

    return model

# ========== STEP 5: Main Training Pipeline ==========
def main():
    print("🚀 Starting Enhanced XGBoost Email Classifier Training...")
    df = load_data()
    df = add_features(df)
    
    X_tfidf, vectorizer = vectorize_text(df["text"])
    structured_features = [
    'email_length', 'num_questions', 'num_exclamations',
    'contains_table', 'has_po', 'has_gst', 'has_model',
    'has_urgent', 'has_project', 'has_specs', 'has_legal',
    'strong_complaint_hits', 'has_negative_tone',
    'has_pricing_word', 'has_certification_request'  # ← NEW
    ]


    structured = df[structured_features].values
    X_combined = hstack([X_tfidf, structured])
    y = df["category"]
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    model = train_xgboost(X_combined, y_encoded, label_encoder, df)
    
    joblib.dump(label_encoder, "label_encoder.joblib")
    joblib.dump(model, "xgboost_email_classifier_no_feedback.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    
    print("\n✅ Model saved as xgboost_email_classifier.joblib")
    print("✅ Vectorizer saved as tfidf_vectorizer.joblib")
    print("✅ Label encoder saved as label_encoder.joblib")

if __name__ == "__main__":
    main()
