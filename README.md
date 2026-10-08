# AI HealthCare Assistant — Advanced Major Project
Modules: Explainable ML triage, emergency safety layer, patient login, dashboard, vitals timeline, appointment workflow, report text analyzer, alerts, chatbot API and Render deployment.
The ML engine uses synthetic educational examples and predicts a care-pathway category, NOT a disease diagnosis.
Run on Windows:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
Open http://127.0.0.1:5000
Render build: pip install -r requirements.txt
Render start: gunicorn app:app

\n## NEW: PDF Medical Report Upload
The Report Analyzer now accepts `.pdf` files (up to 10 MB), extracts text with `pypdf`, detects configured warning phrases, shows page count and a text preview, and stores the extracted report text in the user's history.
For scanned/image-only PDFs, the app reports that OCR is required; this version does not pretend to interpret images.

## Nearby Doctor Finder (integrated)
The same project includes location + disease/symptom based nearby provider search. Enter symptoms and a city/area, or allow GPS. The app classifies the recommended specialist and queries OpenStreetMap/Overpass for nearby mapped doctors, clinics and hospitals, with distance and Maps links. Provider data and availability may be incomplete.
