import pandas as pd
from pathlib import Path
import os
from typing import List, Optional
import pdfplumber
import re
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util


class LocateFile:
    def __init__(self, folder: Path) -> None:
        self.folder = folder

    @staticmethod
    def find_file(folder: Path, file: Path) -> Optional[Path]:
        for obj in os.listdir(folder):
            obj_path = folder / obj
            if str(file) == str(obj):
                return obj_path
            if obj_path.is_dir():
                search = LocateFile.find_file(obj_path, file)
                if search:
                    return search
        return None

    def find_files(self, files: List[Path]) -> List[Optional[Path]]:
        return [self.find_file(self.folder, file) for file in files]


class ExtractAmount:
    def __init__(self, model="textgain/allnli-GroNLP-bert-base-dutch-cased"):
        self.amount_pattern = re.compile(r"€\s*(\d+(?:\.\d+)?)")
        self.model = SentenceTransformer(model)

    @staticmethod
    def clean_text(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def normalize_description(text: str) -> str:
        text = ExtractAmount.clean_text(text)
        replacements = {
            "wilverband": "wildverband",
            "doorstrijken": "doorstrijk",
            "vermetselen": "metselen",
            "gevelsteen": "gevelsteen",
            "accentsteen": "accentsteen",
            "strooisteen": "strooisteen",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"[^a-z0-9\s-]", " ", text)
        text = " ".join(text.split())
        return text

    @staticmethod
    def extract_lines(pdf_path: Path) -> List[str]:
        lines = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                for line in text.split("\n"):
                    line = ExtractAmount.clean_text(line)
                    if line:
                        lines.append(line)
        return lines

    def parse_price_lines(self, pdf_path: Path) -> List[dict]:
        """
        Parse regels zoals:
        'toeslag wildverband € 45.00 dznd'
        """
        lines = self.extract_lines(pdf_path)
        records = []

        pattern = re.compile(
            r"^(?P<desc>.*?)\s+€\s*(?P<amount>\d+(?:\.\d+)?)\s+(?P<unit>[a-z0-9]+)\s*$"
        )

        for line in lines:
            m = pattern.match(line)
            if not m:
                continue
            desc = m.group("desc").strip()
            amount = float(m.group("amount"))
            unit = m.group("unit").strip()

            # headers en ruis uitsluiten
            if len(desc) < 3:
                continue
            if desc in {"omschrijving prijs eenheid", "metselwerk:"}:
                continue

            records.append(
                {
                    "raw_line": line,
                    "description": desc,
                    "description_norm": self.normalize_description(desc),
                    "amount": amount,
                    "unit": unit,
                }
            )
        return records

    @staticmethod
    def fuzzy_score(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def token_overlap_score(a: str, b: str) -> float:
        a_tokens = set(a.split())
        b_tokens = set(b.split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    def find_amount_query(self, query: str, file_path: Path):
        records = self.parse_price_lines(file_path)
        if not records:
            return None, None

        query_norm = self.normalize_description(query)

        # 1. rule-based / fuzzy baseline
        for rec in records:
            rec["fuzzy"] = self.fuzzy_score(query_norm, rec["description_norm"])
            rec["token_overlap"] = self.token_overlap_score(query_norm, rec["description_norm"])

        # 2. embeddings alleen op descriptions
        descriptions = [r["description_norm"] for r in records]
        desc_emb = self.model.encode(descriptions, convert_to_tensor=True)
        query_emb = self.model.encode([query_norm], convert_to_tensor=True)
        sim = util.cos_sim(query_emb, desc_emb)[0].tolist()

        for rec, s in zip(records, sim):
            rec["semantic"] = float(s)
            rec["score"] = 0.45 * rec["fuzzy"] + 0.25 * rec["token_overlap"] + 0.30 * rec["semantic"]

        best = max(records, key=lambda x: x["score"])
        return best["amount"], best

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")

def main():
    data_path = Path("dataset.csv")
    df = load_data(data_path)

    files = list(df["Filename"])
    locater = LocateFile(Path("../../../data"))
    file_paths = locater.find_files(files)
    df["Filepath"] = file_paths

    extractor = ExtractAmount()

    results = []
    for _, row in df.iterrows():
        amount, best = extractor.find_amount_query(row["Post"], row["Filepath"])
        results.append(
            {
                "Post": row["Post"],
                "ExpectedAmount": row["Amount"],
                "FoundAmount": amount,
                "MatchedDescription": None if best is None else best["description"],
                "MatchedLine": None if best is None else best["raw_line"],
                "Score": None if best is None else best["score"],
            }
        )

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    main()