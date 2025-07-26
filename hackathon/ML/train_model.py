import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Sample training data (you can expand this later)
data = [
    ("I need a quotation for fire extinguishers.", "quotation_request"),
    ("Please send me the price for ABC item.", "quotation_request"),
    ("The extinguisher is not working. Please resolve it.", "complaint"),
    ("I’m disappointed with the damaged product I received.", "complaint"),
    ("Any update on our order?", "follow_up"),
    ("Just wanted to say thank you for the support.", "feedback"),
    ("The PO is approved. Please proceed.", "sales_approval"),
    ("Your shipment has been dispatched via FedEx.", "dispatch_update"),
    ("Can I get more information about your services?", "general_enquiry")
]

df = pd.DataFrame(data, columns=["text", "category"])

# Step 2: Split data (optional for small data)
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["category"], test_size=0.2, random_state=42)

# Step 3: Create pipeline
vectorizer = TfidfVectorizer()
model = LogisticRegression()
X_train_vec = vectorizer.fit_transform(X_train)
model.fit(X_train_vec, y_train)

# Step 4: Save model and vectorizer separately
joblib.dump({
    "model": model,
    "vectorizer": vectorizer
}, "spam_detection_model.joblib")

# Optional: Evaluate
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
print("✅ Model trained and saved as spam_detection_model.joblib")
