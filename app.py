from flask import Flask, request, render_template
import pdfplumber
import spacy
import re
from docx import Document
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import pytesseract

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")  # Load the spaCy model
st_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load semantic model

# Skill keywords for extraction
skill_keywords = [
    "Python", "JavaScript", "Machine Learning", "Data Science", "SQL", 
    "Cloud Computing", "React", "Node.js", "Flask", "AI"
]

def extract_text_from_pdf(file):
    """Extracts text from PDF file."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_word(file):
    """Extracts text from Word file."""
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_image(file):
    """Extracts text from image file using OCR."""
    img = Image.open(file)
    return pytesseract.image_to_string(img)

def extract_entities(text):
    """Extract entities using NLP and regex."""
    doc = nlp(text)
    entities = {
        "Name": None,
        "Email": None,
        "Phone Number": None,
        "Skills": [],
        "Profile": None,
    }

    # Extract Name
    person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if person_names:
        entities["Name"] = person_names[0]

    # Extract Email
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    emails = email_pattern.findall(text)
    if emails:
        entities["Email"] = emails[0]

    # Extract Phone Number
    phone_pattern = re.compile(r'(\+?\d{1,3})?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
    phones = phone_pattern.findall(text)
    if phones:
        entities["Phone Number"] = ''.join(phones[0])

    # Extract Skills (Semantic)
    text_embedding = st_model.encode(text, convert_to_tensor=True)
    skills_embedding = st_model.encode(skill_keywords, convert_to_tensor=True)
    results = util.semantic_search(text_embedding, skills_embedding, top_k=5)
    entities["Skills"] = [skill_keywords[match['corpus_id']] for match in results[0]]

    # Extract Profile
    profile_keywords = ["Profile", "Summary", "Objective"]
    for keyword in profile_keywords:
        match = re.search(rf'{keyword}[:\n\s]+(.+?)(\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if match:
            entities["Profile"] = match.group(1).strip()
            break

    return entities

@app.route("/", methods=["GET", "POST"])
def upload_resume():
    if request.method == "POST":
        file = request.files["resume"]
        if file:
            file_ext = file.filename.split('.')[-1].lower()
            text = ""

            # Text extraction based on file type
            if file_ext == "pdf":
                text = extract_text_from_pdf(file)
            elif file_ext == "docx":
                text = extract_text_from_word(file)
            elif file_ext in ["png", "jpg", "jpeg"]:
                text = extract_text_from_image(file)
            else:
                return "Unsupported file format. Please upload PDF, Word, or image files."

            entities = extract_entities(text)
            return render_template("results.html", text=text[:1000], entities=entities)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

