from pathlib import Path
import re
import pdfplumber
from sentence_transformers import SentenceTransformer, util

haan = Path(
    "../../data/03.02_Offertes_ontvangen/38.00_gevelschermen/Haan/Offerte 10054419-1 Van Wijnen Gorredijk BV (1).pdf"
)

KEYWORDS = [
    "totaal",
    "totaalbedrag",
    "aanneemsom",
    "offertebedrag",
    "eindtotaal",
    "subtotaal",
    "totaal excl",
    "totaal incl",
    "excl. btw",
    "incl. btw",
]

QUERIES = [
    "totaalbedrag",
    "totale aanneemsom",
    "offertebedrag",
    "eindtotaal",
    "totaal exclusief btw",
    "totaal inclusief btw",
]

AMOUNT_PATTERN = re.compile(r"(€\s*)?\d{1,3}(?:\.\d{3})*,\d{2}|\b\d+,\d{2}\b")


def clean_text(text: str) -> str:
    return " ".join(text.split())


def extract_lines(pdf_path: Path) -> list[str]:
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                line = clean_text(line)
                if line:
                    lines.append(line)
    return lines


def make_windows(lines: list[str], window_sizes=(1, 2, 3)) -> list[dict]:
    windows = []
    for w in window_sizes:
        for i in range(len(lines) - w + 1):
            chunk = " ".join(lines[i:i+w])
            windows.append({
                "start": i,
                "end": i + w,
                "text": chunk,
            })
    return windows


def find_amounts(text: str) -> list[str]:
    return AMOUNT_PATTERN.findall(text)


def has_amount(text: str) -> bool:
    return re.search(AMOUNT_PATTERN, text) is not None


def keyword_score(text: str) -> int:
    lower = text.lower()
    return sum(1 for kw in KEYWORDS if kw in lower)


def extract_first_amount(text: str) -> str | None:
    match = re.search(AMOUNT_PATTERN, text)
    return match.group(0) if match else None


lines = extract_lines(haan)
windows = make_windows(lines, window_sizes=(1, 2, 3))

# Eerst filteren op relevante kandidaten
candidates = []
for w in windows:
    kw_score = keyword_score(w["text"])
    amount_present = has_amount(w["text"])

    # Kandidaten moeten of een keyword hebben, of een bedrag + relevante context
    if kw_score > 0 or amount_present:
        candidates.append({
            **w,
            "keyword_score": kw_score,
            "amount_present": amount_present,
        })

print(f"Aantal kandidaten: {len(candidates)}")

model = SentenceTransformer("textgain/allnli-GroNLP-bert-base-dutch-cased")

candidate_texts = [c["text"] for c in candidates]
candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)
query_embeddings = model.encode(QUERIES, convert_to_tensor=True)

sim_matrix = util.cos_sim(query_embeddings, candidate_embeddings)

results = []
for i, candidate in enumerate(candidates):
    best_sim = float(sim_matrix[:, i].max())
    amount_bonus = 0.25 if candidate["amount_present"] else 0.0
    keyword_bonus = 0.10 * candidate["keyword_score"]

    total_score = best_sim + amount_bonus + keyword_bonus

    results.append({
        **candidate,
        "semantic_score": best_sim,
        "total_score": total_score,
        "amount": extract_first_amount(candidate["text"]),
    })

results = sorted(results, key=lambda x: x["total_score"], reverse=True)

for r in results[:15]:
    print("\n---")
    print(f"score={r['total_score']:.4f} | semantic={r['semantic_score']:.4f} | kw={r['keyword_score']} | amount={r['amount_present']}")
    print(r["text"])