from flask import Flask, request, render_template, flash
import joblib
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv
from spam_classifier import clean_text as training_clean_text
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from config.env file
load_dotenv('config.env')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))

# Load model and vectorizer
try:
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    logger.info("ML models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    model = None
    vectorizer = None

# Gemini setup
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel("gemini-2.0-flash")
    logger.info("Gemini API configured successfully")
else:
    logger.warning("No Gemini API key found - response generation will be disabled")
    model_gemini = None

def clean_text(text):
    """Use the same cleaning as training for consistent inference."""
    return training_clean_text(text)

# Classification tuning
SPAM_THRESHOLD = float(os.getenv('SPAM_THRESHOLD', '0.45'))  # classify as spam if P(spam) >= threshold
RULES_STRICT = os.getenv('RULES_STRICT', 'true').lower() == 'true'

SPAM_CUE_PATTERNS = [
    r"\bfree\b",
    r"\bwinner\b|\bwon\b|\bwin\b",
    r"claim\s+now",
    r"click\s+here",
    r"congratulations",
    r"\bprize\b|\bjackpot\b",
    r"urgent|immediately|act now",
    r"http[s]?://",
    r"\b(?:\w+[-.]){1,}\w+\.(?:biz|info|download)\b",
    r"\btext\s+[A-Z]{2,}\b",
    r"\bsubscription\b|\bsusbscribe\b|\bopt\s*out\b",
]

def detect_spam_cues(raw_text: str):
    text = (raw_text or "")
    text_lower = text.lower()
    matches = []
    for pat in SPAM_CUE_PATTERNS:
        if re.search(pat, text_lower):
            matches.append(pat)
    return len(matches), matches

def generate_email_response(email_text):
    """Generate a response to an email using the Gemini API."""
    if not GOOGLE_API_KEY or not model_gemini:
        return "API key not configured. Please set GOOGLE_API_KEY in environment variables."
    
    prompt = (
        f"You are an AI assistant. Read the following email:\n\n"
        f"\"{email_text}\"\n\n"
        f"Write a clear, polite, and professional response that addresses the main points of the email. "
        f"Keep it concise and friendly."
    )
    
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating response with Gemini: {e}")
        return f"Error generating response: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route handler for the spam classifier application."""
    result = None
    ai_response = None
    confidence_score = None

    if request.method == 'POST':
        email_text = request.form.get('email', '')
        raw_text = (email_text or '').strip()
        
        # Basic empty/length validation
        if not raw_text:
            flash("Please enter email content to analyze")
            return render_template('index.html')
        if len(raw_text) < 3:
            flash("Email content is too short to analyze")
            return render_template('index.html')
        
        if not model or not vectorizer:
            flash("ML models not loaded. Please check server logs.")
            return render_template('index.html')
            
        try:
            # Clean and validate meaningful content
            cleaned = clean_text(raw_text)
            if not cleaned or len(cleaned.strip()) == 0:
                flash("Email content contains no meaningful text after cleaning")
                return render_template('index.html')
            
            # Classify email
            vectorized = vectorizer.transform([cleaned])

            # Probability-based decision with threshold
            proba = model.predict_proba(vectorized)[0]
            classes = list(getattr(model, 'classes_', [0, 1]))
            try:
                spam_idx = classes.index(1)
                ham_idx = classes.index(0)
            except ValueError:
                # Fallback assumption 0=ham,1=spam
                spam_idx, ham_idx = 1, 0

            p_spam = float(proba[spam_idx])
            p_ham = float(proba[ham_idx])

            result = "Spam" if p_spam >= SPAM_THRESHOLD else "Not Spam"
            confidence_score = round((p_spam if result == "Spam" else p_ham) * 100, 2)

            # Rule-based override for strong cues
            if RULES_STRICT:
                cues_count, cues = detect_spam_cues(email_text)
                if cues_count >= 2 and result == "Not Spam":
                    logger.info(f"Rule override to Spam due to cues: {cues}")
                    result = "Spam"
                    confidence_score = max(confidence_score, round(p_spam * 100, 2), 90.0)

            logger.info(f"Classification result: {result} (P_spam={p_spam:.3f}, threshold={SPAM_THRESHOLD}, confidence={confidence_score}%)")

            # Generate response for non-spam emails
            if result == "Not Spam":
                ai_response = generate_email_response(email_text)
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            flash(f"Error during prediction: {str(e)}")
            
    return render_template('index.html', 
                          result=result, 
                          ai_response=ai_response,
                          confidence_score=confidence_score)

if __name__ == '__main__':
    # Get configuration from environment
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    
    # Check if models are loaded
    if not model or not vectorizer:
        logger.warning("Warning: ML models could not be loaded. Application may not function correctly.")
    
    # Run the Flask app
    logger.info(f"Starting application on port {port}, debug mode: {debug_mode}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
