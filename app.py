# app.py

from flask import Flask, request, render_template_string
import torch
from inference import load_model, analyze_text

app = Flask(__name__)

# Load model once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, word2idx = load_model("sentiment_model.pt", device=device)

# A simple HTML template using Flask's render_template_string
template = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis App</title>
</head>
<body>
    <h1>Enter Text for Sentiment Analysis</h1>
    <form method="POST" action="/">
        <textarea name="input_text" rows="6" cols="60">{{ input_text }}</textarea><br><br>
        <input type="submit" value="Analyze">
    </form>
    
    {% if sentences %}
    <h2>Results</h2>
    <div>
        {% for sentence, color, score in sentences %}
            <p style="color: {{color}};">{{ sentence }} (Score: {{ score|round(3) }})</p>
        {% endfor %}
    </div>
    <h3>Overall Sentiment Score: {{ overall_score|round(3) }} ({{ (overall_score * 100)|round(2) }} / 100)</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    sentences = []
    overall_score = 0.0
    
    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        if input_text.strip():
            sents, overall = analyze_text(input_text, model, word2idx, device=device)
            sentences = sents
            overall_score = overall
    
    return render_template_string(
        template,
        input_text=input_text,
        sentences=sentences,
        overall_score=overall_score
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
