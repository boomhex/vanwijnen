from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import os
import pdfplumber
import re

kellegens = Path(
    "../../data/03.02_Offertes_ontvangen/38.00_gevelschermen/Zonwering industrie/25612.A  IKC Sint Nicolaasga.pdf"
)

haan = Path(
    "/Users/timojolman/Zakelijk/UniPartners/vanWijnen/vanwijnen/data/03.02_Offertes_ontvangen/38.00_gevelschermen/Haan/Offerte 10054419-1 Van Wijnen Gorredijk BV (1).pdf"
)

def clean_line(line):
    return ' '.join(line.split())

def extract_lines(path):
    lines = []
    with pdfplumber.open(haan) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_lines = text.split('\n')
            for line in page_lines:
                lines.append(clean_line(line))
    return lines

lines = extract_lines(haan)
filtered_lines = []

for line in lines:
    if "€" in line:
        filtered_lines.append(line)

def find_first_amount(text: str) -> str | None:
    # Voor NL bedragen zoals:
    # 1.234,56
    # € 1.234,56
    # 1234,56
    pattern = r"(€\s*)?\d{1,3}(?:\.\d{3})*,\d{2}|\b\d+,\d{2}\b"
    match = re.search(pattern, text)
    return match.group(0) if match else None


model = SentenceTransformer("textgain/allnli-GroNLP-bert-base-dutch-cased")

queries = [
    "Totaalbedrag",
    "aanneemsom",
    "eindtotaal",
    "totaal inclusief btw",
    "totaal exclusief btw",
]

def find_semantic_line(queries, lines):
    line_embeddings = model.encode(lines, convert_to_tensor=True)
    query_embeddings = model.encode(queries, convert_to_tensor=True)

    scores = util.cos_sim(query_embeddings, line_embeddings)
    best_matches = []
    for q_idx, query in enumerate(queries):
        best_idx = int(scores[q_idx].argmax())
        best_score = float(scores[q_idx][best_idx])
        best_line = lines[best_idx]
        amount = find_first_amount(best_line)

        best_matches.append({
            "query": query,
            "score": best_score,
            "line": best_line,
            "amount": amount,
        })

    best_matches = sorted(best_matches, key=lambda x: x["score"], reverse=True)

    for match in best_matches[:5]:
        print(f"\nQuery: {match['query']}")
        print(f"Score: {match['score']:.4f}")
        print(f"Line: {match['line']}")
        print(f"Amount: {match['amount']}")

find_semantic_line(queries, filtered_lines)
