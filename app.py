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
# Serve static assets from the 'assets' folder at URL path '/assets'
app = Flask(__name__, static_folder='assets', static_url_path='/assets')
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
    r"\bfree\b.*\b(?:offer|gift|prize|money)\b",  # Free + valuable item
    r"\bwinner\b|\bwon\b.*\b(?:prize|money|gift)\b",  # Winner context
    r"claim\s+(?:now|immediately|today)",
    r"click\s+here.*(?:now|immediately)",
    r"congratulations.*(?:won|selected|winner)",
    r"\bprize\b|\bjackpot\b",
    r"urgent.*(?:immediately|act now|respond now)",
    r"(?:verify|confirm|update).*(?:immediately|now|today)",
    r"\b(?:password|account|security).*(?:expired|suspended|locked)",
    r"hxx?p://",  # Obfuscated HTTP
    r"replace\s+\w+\s+with\s+\w+",  # URL obfuscation instruction
    r"\b(?:\w+[-.]){2,}\w+\.(?:biz|info|download|tk|ml)\b",  # Suspicious domains
    r"\btext\s+[A-Z]{2,}\s+to\s+\d+",  # SMS spam pattern
    r"\b(?:ssn|social security|credit card|bank account)\b.*(?:provide|enter|confirm)",  # Info harvesting
    r"mailbox.*(?:full|quota|exceeded)",  # Phishing theme
    r"re-?validate.*(?:account|email|identity)",  # Phishing validation
]

def detect_spam_cues(raw_text: str):
    text = (raw_text or "")
    text_lower = text.lower()
    
    # Check for legitimate patterns that should reduce spam score
    legitimate_patterns = [
        r"invoice.*(?:paid|due|total)",  # Legitimate invoices
        r"meeting.*(?:tomorrow|scheduled|invite)",  # Meeting emails
        r"re:\s*",  # Reply emails
        r"fwd:\s*",  # Forwarded emails
        r"unsubscribe.*link",  # Legitimate newsletters
        r"github|microsoft|google|amazon\.com",  # Trusted domains in legitimate context
    ]
    
    # Count legitimate indicators
    legit_count = sum(1 for pat in legitimate_patterns if re.search(pat, text_lower))
    
    # Count spam indicators
    spam_matches = []
    for pat in SPAM_CUE_PATTERNS:
        if re.search(pat, text_lower):
            spam_matches.append(pat)
    
    # Reduce spam score if legitimate patterns found
    spam_score = len(spam_matches)
    if legit_count > 0:
        spam_score = max(0, spam_score - legit_count)
    
    return spam_score, spam_matches

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
    enable_ai_flag = True  # default to previous behavior

    if request.method == 'POST':
        email_text = request.form.get('email', '')
        # Handle checkbox + hidden fallback: if any value is 'on', then enabled
        try:
            enable_vals = [str(v).lower() for v in request.form.getlist('enable_ai')]
        except Exception:
            enable_vals = [str(request.form.get('enable_ai', 'on')).lower()]
        enable_ai_flag = any(v in ('on', 'true', '1', 'yes') for v in enable_vals)
        raw_text = (email_text or '').strip()
        
        # Basic empty/length validation
        if not raw_text:
            flash("Please enter email content to analyze")
            return render_template('index.html', enable_ai=enable_ai_flag)
        
        if not model or not vectorizer:
            flash("ML models not loaded. Please check server logs.")
            return render_template('index.html', enable_ai=enable_ai_flag)
            
        try:
            # Clean and validate meaningful content
            cleaned = clean_text(raw_text)
            if not cleaned or len(cleaned.strip()) == 0:
                flash("Email content contains no meaningful text after cleaning")
                return render_template('index.html', enable_ai=enable_ai_flag)
            
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

            # Enhanced rule-based override for strong cues
            if RULES_STRICT:
                cues_count, cues = detect_spam_cues(email_text)
                
                # Strong spam indicators - override even high confidence "Not Spam"
                strong_spam_patterns = [
                    r"hxx?p://",  # Obfuscated URLs
                    r"replace\\s+\\w+\\s+with\\s+\\w+",  # URL obfuscation
                    r"\\b(?:ssn|social security)\\b.*(?:provide|enter)",  # SSN harvesting
                    r"mailbox.*(?:full|quota).*re-?validate",  # O365 phishing pattern
                    r"(?:verify|confirm).*(?:account|identity).*immediately"  # Urgent verification
                ]
                
                has_strong_indicators = any(re.search(pat, email_text.lower()) for pat in strong_spam_patterns)
                
                # Override logic
                if has_strong_indicators and result == "Not Spam":
                    logger.info(f"Strong spam pattern override: {[pat for pat in strong_spam_patterns if re.search(pat, email_text.lower())]}")
                    result = "Spam"
                    confidence_score = max(confidence_score, 85.0)
                elif cues_count >= 2 and result == "Not Spam" and p_spam > 0.3:
                    logger.info(f"Rule override to Spam due to cues: {cues}")
                    result = "Spam"
                    confidence_score = max(confidence_score, round(p_spam * 100, 2), 80.0)
                
                # Reduce false positives for legitimate HTML content
                if "<html>" in email_text.lower() and "invoice" in email_text.lower() and result == "Spam":
                    if not has_strong_indicators and cues_count < 3:
                        logger.info("Legitimate HTML invoice detected, overriding to Not Spam")
                        result = "Not Spam"
                        confidence_score = max(60.0, round(p_ham * 100, 2))

            logger.info(f"Classification result: {result} (P_spam={p_spam:.3f}, threshold={SPAM_THRESHOLD}, confidence={confidence_score}%)")

            # Generate response for non-spam emails if enabled and API available
            if result == "Not Spam" and enable_ai_flag:
                if model_gemini is not None and GOOGLE_API_KEY:
                    ai_response = generate_email_response(email_text)
                else:
                    logger.info("AI response skipped: Gemini API key not configured")
                    flash("AI response is disabled (API key not configured)")
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            flash(f"Error during prediction: {str(e)}")
            
    return render_template('index.html', 
                          result=result, 
                          ai_response=ai_response,
                          confidence_score=confidence_score,
                          enable_ai=enable_ai_flag)

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
