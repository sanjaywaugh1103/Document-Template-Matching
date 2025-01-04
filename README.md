
# Document Template Matching Using OCR

This project focuses on document template classification and matching using OCR (Optical Character Recognition) and pattern-based detection. It leverages AI techniques to identify different types of documents and classify them into appropriate categories.

## Features

- **OCR-Based Text Extraction:** Extracts text from documents using Tesseract OCR.
- **Pattern-Based Detection:** Identifies specific patterns for documents like admission forms and insurance documents.
- **AI Classification:** Utilizes Naive Bayes classifier for document classification based on extracted text.
- **Keyword Matching:** Fallback mechanism for classification using predefined keywords.
- **Automated Folder Organization:** Moves classified documents into respective folders for easy access.

## Categories Supported

1. Admission Forms
2. Aadhaar Cards
3. Lab Reports
4. Medication Reports
5. Insurance Documents

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/document-template-matching.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Tesseract OCR is installed on your system. [Download Tesseract](https://github.com/tesseract-ocr/tesseract).

## Folder Structure

- `templates/`: Contains template documents for AI training.
- `uploaded_documents/`: Place documents for classification here.
- `classified_documents/`: Classified documents will be organized here.
  - `unrecognized/`: Unclassified documents will be stored here.
- `Document_template_Matchingf.py`: Main script for document classification.

## Usage

1. Place your template documents in the `templates/` folder.
2. Place the documents you want to classify in the `uploaded_documents/` folder.
3. Run the script:
   ```bash
   python Document_template_Matchingf.py
   ```
4. Check the `classified_documents/` folder for results.

## Key Functions

- `extract_text`: Extracts text from images using OCR.
- `detect_admission_form`: Detects admission forms based on patterns.
- `detect_insurance_form`: Detects insurance documents based on patterns.
- `train_ai_model`: Trains an AI model for document classification.
- `classify_with_ai`: Classifies documents using the trained AI model.
- `classify_with_keywords`: Classifies documents using keyword matching.
- `classify_document`: Main function for document classification.

## Prerequisites

- Python 3.7 or higher
- OpenCV
- Tesseract OCR
- Scikit-learn
- NumPy

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
