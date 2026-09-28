import os
import pickle
import re
import string
import pandas as pd
from flask import Flask, request, render_template, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading model and vectorizer...")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Model and vectorizer loaded successfully.")

# Sensationalist and clickbait markers
SENSATIONAL_PATTERNS = [
    r'\bbreaking\b', r'\bshocking\b', r'you won\'t believe', r'\bexposed\b',
    r'\bbombshell\b', r'\bsecret\b', r'\bwhistleblower\b', r'\bhoax\b',
    r'\bmiracle cure\b', r'\bdeep state\b', r'\bconspiracy\b', r'\bwake up\b',
    r'\blizard people\b', r'what they don\'t want you to know', r'\bviral\b',
    r'\bproven fake\b', r'\bcover[- ]?up\b', r'\bclone\b',
    r'\bmainstream media hides\b', r'doctors baffled', r'\bshocks world\b',
    r'\billuminati\b', r'\bsecret revealed\b', r'\bshocking truth\b',
    r'\bsecret plot\b', r'\bthey lied\b', r'\btruth revealed\b'
]

# Credibility and objective journalistic markers
CREDIBILITY_PATTERNS = [
    r'\baccording to\b', r'\bofficials? said\b', r'\bannounced\b', r'\bspokes(person|man|woman)\b',
    r'\bstatement\b', r'\bresearchers?\b', r'\bpublished\b', r'\bdepartment\b',
    r'\bministry\b', r'\bconfirmed\b', r'\bauthorities\b', r'\bstudy\b',
    r'\bpercent\b', r'\bquarterly\b', r'\bgovernment\b', r'\bparliament\b',
    r'\bmeeting\b', r'\bagreement\b', r'\bagency\b', r'\bscientists?\b',
    r'\binstitute\b', r'\buniversity\b', r'\btelescope\b', r'\bsatellite\b',
    r'\breports?\b', r'\bjournalists?\b', r'\bofficial\b', r'\bfindings\b',
    r'\binvestigation\b', r'\beconomic\b', r'\bsecurity\b', r'\bhealth\b',
    r'\bspokesman said\b', r'\bpress release\b', r'\bspokesperson\b'
]

def wordopt(text):
    """Cleans and preprocesses the input text."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def evaluate_news(raw_text):
    """Evaluates whether news text is reliable or fake using ensemble ML + stylistic credibility analysis."""
    cleaned_text = wordopt(raw_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # ML Ensemble probability for class 1 (Reliable)
    ml_prob = float(model.predict_proba(vectorized_text)[0][1])
    
    words = raw_text.split()
    total_words = len(words)
    total_chars = len(raw_text)
    
    # Known legitimate uppercase acronyms
    safe_acronyms = {'USA', 'NASA', 'NATO', 'WHO', 'UN', 'GDP', 'FBI', 'CIA', 'EU', 'UK', 'US', 'IMF', 'CEO', 'AI', 'BBC', 'CNN', 'AP', 'ISRO'}
    caps_words = [w for w in words if w.isupper() and len(w) > 2 and w not in safe_acronyms]
    caps_ratio = len(caps_words) / max(total_words, 1)
    multi_punc = len(re.findall(r'!{2,}|\?{2,}|\?!|!\?', raw_text))
    
    # Find matching phrases
    sens_found = []
    for p in SENSATIONAL_PATTERNS:
        for m in re.finditer(p, raw_text, re.IGNORECASE):
            sens_found.append(m.group(0))
            
    cred_found = []
    for p in CREDIBILITY_PATTERNS:
        for m in re.finditer(p, raw_text, re.IGNORECASE):
            cred_found.append(m.group(0))
            
    sens_matches = list(dict.fromkeys(sens_found))
    cred_matches = list(dict.fromkeys(cred_found))
    
    # Penalties and bonuses
    penalty = len(sens_matches) * 0.18 + (0.15 if caps_ratio > 0.15 else 0) + (0.15 if multi_punc > 0 else 0)
    bonus = min(len(cred_matches) * 0.10, 0.30) if penalty == 0 else 0
    
    adjusted_prob = ml_prob + bonus - penalty
    
    # Prevent factual non-political statements with zero sensationalism from false fake flagging
    if penalty == 0 and len(cred_matches) > 0 and adjusted_prob < 0.50:
        adjusted_prob = max(adjusted_prob + 0.35, 0.55)
        
    adjusted_prob = max(0.01, min(0.99, adjusted_prob))
    
    is_reliable = adjusted_prob >= 0.50
    trust_score = int(round(adjusted_prob * 100))
    confidence = int(round((adjusted_prob if is_reliable else (1.0 - adjusted_prob)) * 100))
    
    # Risk and category metrics
    if len(sens_matches) >= 2 or caps_ratio > 0.2:
        clickbait_risk = "High"
    elif len(sens_matches) == 1 or caps_ratio > 0.1 or multi_punc > 0:
        clickbait_risk = "Moderate"
    else:
        clickbait_risk = "Low"
        
    tone = "Sensational / Emotionally Charged" if clickbait_risk != "Low" else "Objective / Factual"
    attribution = "Verifiable Sources / Citations Present" if len(cred_matches) > 0 else "Neutral / Standard Reporting"
    
    if is_reliable:
        verdict_summary = "This article exhibits neutral, fact-oriented journalistic characteristics with low clickbait propensity."
    else:
        verdict_summary = "This text exhibits patterns frequently associated with misinformation, sensationalism, or unsubstantiated claims."

    return {
        'is_reliable': is_reliable,
        'trust_score': trust_score,
        'confidence': confidence,
        'clickbait_risk': clickbait_risk,
        'tone': tone,
        'attribution': attribution,
        'verdict_summary': verdict_summary,
        'sens_matches': sens_matches,
        'cred_matches': cred_matches,
        'caps_count': len(caps_words),
        'multi_punc_count': multi_punc,
        'word_count': total_words,
        'char_count': total_chars
    }

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives news text, evaluates veracity, and returns structured analysis JSON."""
    if request.method == 'POST':
        news_text = request.form.get('news_text', '')

        if not news_text.strip():
            return jsonify({'error': 'Input text cannot be empty. Please enter news text or headline.'}), 400

        analysis = evaluate_news(news_text)

        if analysis['is_reliable']:
            result = "This looks like a Reliable News source."
            label = "reliable"
        else:
            result = "This appears to be Fake News."
            label = "fake"

        return jsonify({
            'prediction_text': result,
            'label': label,
            'trust_score': analysis['trust_score'],
            'confidence': analysis['confidence'],
            'clickbait_risk': analysis['clickbait_risk'],
            'tone': analysis['tone'],
            'attribution': analysis['attribution'],
            'verdict_summary': analysis['verdict_summary'],
            'cred_matches': analysis['cred_matches'],
            'sens_matches': analysis['sens_matches'],
            'caps_count': analysis['caps_count'],
            'multi_punc_count': analysis['multi_punc_count'],
            'word_count': analysis['word_count'],
            'char_count': analysis['char_count']
        })

if __name__ == '__main__':
    app.run(debug=True)
