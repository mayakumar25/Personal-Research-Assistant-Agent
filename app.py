from flask import Flask, render_template, request
import requests
from transformers import pipeline

app = Flask(__name__)

# Load summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Semantic Scholar API
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def fetch_papers(query, limit=5):
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,url,year"
    }
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return []

def summarize_text(text):
    if not text:
        return "No abstract available."
    if len(text.split()) < 50:
        return text  # Skip summarization for very short abstracts
    try:
        result = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        print(f"Summarization failed: {e}")
        return "Summary unavailable due to processing error."

@app.route("/", methods=["GET", "POST"])
def index():
    summary_results = []
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            papers = fetch_papers(query)
            for paper in papers:
                summary = summarize_text(paper.get("abstract", ""))
                summary_results.append({
                    "title": paper.get("title", "No title"),
                    "authors": ", ".join([a.get("name", "Unknown") for a in paper.get("authors", [])]),
                    "year": paper.get("year", "N/A"),
                    "url": paper.get("url", "#"),
                    "summary": summary
                })
    return render_template("index.html", results=summary_results)

if __name__ == "__main__":
    app.run(debug=False)
