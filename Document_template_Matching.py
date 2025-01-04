import cv2
import os
import shutil
import pytesseract
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths
TEMPLATE_FOLDER = "templates/"
OUTPUT_FOLDER = "classified_documents/"
UNRECOGNIZED_FOLDER = os.path.join(OUTPUT_FOLDER, "unrecognized")
os.makedirs(UNRECOGNIZED_FOLDER, exist_ok=True)

# Define categories and keywords
TEMPLATE_TYPES = ['admission_form', 'aadhaar_card', 'lab_report', 'medication_report', 'insurance_document']
KEYWORDS = {
    'admission_form': [
        'admission', 'form', 'patient', 'hospital', 'date of admission',
        'discharge date', 'doctor', 'ward', 'admitted on', 'admitted to', 'discharged on'
    ],
    'aadhaar_card': ['aadhaar', 'uidai', 'identity', 'dob'],
    'lab_report': ['blood group', 'bp', 'cholesterol', 'lab', 'test report'],
    'medication_report': ['medication', 'prescription', 'doctor', 'dose', 'handwritten'],
    'insurance_document': ['insurance', 'policy', 'premium', 'claim', 'coverage', 'insured', 'policy number']
}

# Data for AI model
documents = []  # Holds text extracted from templates
labels = []  # Holds corresponding template types

# Preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read image at {image_path}.")
        return None
    image = cv2.resize(image, (500, 500))
    return image

# Extract text using OCR
def extract_text(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}.")
        return ""
    text = pytesseract.image_to_string(image)
    print(f"Extracted Text: {text}")
    return text

# Pattern-based detection for admission forms
def detect_admission_form(document_text):
    patterns = [
        r"(date of admission[:\s]*\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"(discharge date[:\s]*\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"(patient name[:\s]*[a-zA-Z\s]+)",
        r"(hospital name[:\s]*[a-zA-Z\s]+)",
        r"(ward[:\s]*\w+)"
    ]
    for pattern in patterns:
        if re.search(pattern, document_text, re.IGNORECASE):
            print(f"Admission Form Pattern Matched: {pattern}")
            return True
    return False

# Pattern-based detection for insurance forms
def detect_insurance_form(document_text):
    patterns = [
        r"(policy number[:\s]*\w+)",  # Detects "Policy Number"
        r"(claim amount[:\s]*\d+)",  # Detects "Claim Amount"
        r"(insured name[:\s]*[a-zA-Z\s]+)",  # Detects "Insured Name"
        r"(premium[:\s]*\d+)"  # Detects "Premium"
    ]
    for pattern in patterns:
        if re.search(pattern, document_text, re.IGNORECASE):
            print(f"Insurance Form Pattern Matched: {pattern}")
            return True
    return False

# Train AI Model
def train_ai_model():
    print("Training AI model...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    y = np.array(labels)

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    print("Model Training Completed.")
    print(classification_report(y_test, predictions))

    return model, vectorizer

# AI-based classification
def classify_with_ai(document_text, model, vectorizer):
    features = vectorizer.transform([document_text])
    predicted_class = model.predict(features)[0]
    confidence = model.predict_proba(features).max()
    print(f"AI Classification: {predicted_class} (Confidence: {confidence})")
    return predicted_class if confidence > 0.6 else None

# Fallback OCR-based classification
def classify_with_keywords(document_text):
    for category, keyword_list in KEYWORDS.items():
        if any(keyword.lower() in document_text.lower() for keyword in keyword_list):
            print(f"Matched category using OCR: {category}")
            return category
    return None

# Classify document
def classify_document(document_path, model, vectorizer):
    print(f"Processing: {document_path}")

    # Step 1: Extract Text
    document_text = extract_text(document_path)

    # Step 2: Pattern Detection for Insurance
    if detect_insurance_form(document_text):
        print(f"Document {document_path} classified as insurance_document based on patterns.")
        save_to_folder('insurance_document', document_path)
        return

    # Step 3: Pattern Detection for Admission Forms
    if detect_admission_form(document_text):
        print(f"Document {document_path} classified as admission_form based on patterns.")
        save_to_folder('admission_form', document_path)
        return

    # Step 4: AI Classification
    ai_classification = classify_with_ai(document_text, model, vectorizer)

    # Step 5: Fallback to Keywords
    if ai_classification:
        save_to_folder(ai_classification, document_path)
    else:
        keyword_classification = classify_with_keywords(document_text)
        if keyword_classification:
            save_to_folder(keyword_classification, document_path)
        else:
            print(f"Document {document_path} could not be classified. Moving to 'unrecognized'.")
            shutil.move(document_path, os.path.join(UNRECOGNIZED_FOLDER, os.path.basename(document_path)))

# Save document to appropriate folder
def save_to_folder(category, document_path):
    target_folder = os.path.join(OUTPUT_FOLDER, category)
    os.makedirs(target_folder, exist_ok=True)
    shutil.move(document_path, os.path.join(target_folder, os.path.basename(document_path)))

# Load templates for training AI/ML
def load_templates():
    for template_name in os.listdir(TEMPLATE_FOLDER):
        template_path = os.path.join(TEMPLATE_FOLDER, template_name)
        document_text = extract_text(template_path)
        template_type = template_name.split('_')[0]  # Example: admission_form_1.jpg
        documents.append(document_text)
        labels.append(template_type)

# Main
if __name__ == "__main__":
    # Load templates and train AI model
    load_templates()
    ai_model, text_vectorizer = train_ai_model()

    # Classify uploaded documents
    uploaded_documents_path = "uploaded_documents/"
    for document in os.listdir(uploaded_documents_path):
        document_path = os.path.join(uploaded_documents_path, document)
        if os.path.isfile(document_path):
            classify_document(document_path, ai_model, text_vectorizer)

    print("Classification complete. Check the 'classified_documents' and 'unrecognized' folders.")
