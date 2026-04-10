from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber
import pandas as pd
import re


class OfferLineExtractor:
    def __init__(self) -> None:
        # Ondersteunt o.a.:
        # 800.00
        # 12.50
        # 6000.00
        # € 53.800,=
        # 53.800,=
        # 1.234,56
        self.amount_pattern = re.compile(
            r"^(?:€\s*)?(?:\d{1,3}(?:[.\s]\d{3})*(?:,\d{2}|,=)?|\d+(?:[.,]\d{2}|,=)?)$"
        )

        # Eenheden die vaak voorkomen in offertes
        self.unit_candidates = {
            "st", "m1", "m2", "m3", "mu", "uur", "post", "dznd", "pst", "kg", "ton"
        }

        # Regels die vaak geen offertepost zijn
        self.stopwords = {
            "omschrijving", "prijs", "eenheid",
            "bijzonderheden", "geachte", "betreft",
            "offertenummer", "project", "behandeld",
            "betalingstermijn", "met vriendelijke groeten"
        }

    @staticmethod
    def clean_text(text: str) -> str:
        return " ".join(text.split()).strip()

    @staticmethod
    def group_words_into_lines(words: List[Dict], y_tolerance: float = 3.0) -> List[List[Dict]]:
        if not words:
            return []

        words = sorted(words, key=lambda w: (w["top"], w["x0"]))
        lines: List[List[Dict]] = []
        current_line = [words[0]]
        current_top = words[0]["top"]

        for word in words[1:]:
            if abs(word["top"] - current_top) <= y_tolerance:
                current_line.append(word)
            else:
                lines.append(sorted(current_line, key=lambda w: w["x0"]))
                current_line = [word]
                current_top = word["top"]

        lines.append(sorted(current_line, key=lambda w: w["x0"]))
        return lines

    def is_amount_token(self, token: str) -> bool:
        token = token.strip()
        return bool(self.amount_pattern.match(token))

    def normalize_amount(self, token: str) -> Optional[float]:
        token = token.replace("€", "").replace(" ", "").replace("=", "").strip()
        if not token:
            return None

        # Nederlandse notatie
        if "," in token:
            token = token.replace(".", "").replace(",", ".")
        return float(token)

    def is_probable_unit(self, token: str) -> bool:
        return token.lower() in self.unit_candidates

    def is_noise_line(self, text: str) -> bool:
        lower = text.lower()
        if len(lower) < 4:
            return True

        for sw in self.stopwords:
            if sw in lower:
                return True

        # geen bedrag = meestal geen prijsregel
        if not any(self.is_amount_token(tok) for tok in lower.split()):
            return True

        return False

    def parse_line(self, line_words: List[Dict]) -> Optional[Dict]:
        tokens = [w["text"] for w in line_words]
        full_text = self.clean_text(" ".join(tokens))

        if self.is_noise_line(full_text):
            return None

        # Zoek eerste bedrag in de regel
        amount_idx = None
        for i, tok in enumerate(tokens):
            # Soms staat € los van bedrag
            if tok == "€" and i + 1 < len(tokens) and self.is_amount_token(tokens[i + 1]):
                amount_idx = i
                break
            if self.is_amount_token(tok):
                amount_idx = i
                break

        if amount_idx is None:
            return None

        # Bedrag reconstrueren
        if tokens[amount_idx] == "€":
            if amount_idx + 1 >= len(tokens):
                return None
            raw_amount = f"{tokens[amount_idx]} {tokens[amount_idx + 1]}"
            amount_end_idx = amount_idx + 1
        else:
            raw_amount = tokens[amount_idx]
            amount_end_idx = amount_idx

        try:
            amount_value = self.normalize_amount(raw_amount)
        except Exception:
            return None

        # Alles vóór het bedrag = omschrijving
        desc_tokens = tokens[:amount_idx]

        # Alles ná het bedrag = meestal eenheid of extra tekst
        trailing_tokens = tokens[amount_end_idx + 1:]

        if not desc_tokens:
            return None

        description = self.clean_text(" ".join(desc_tokens))
        unit = None
        rest = None

        if trailing_tokens:
            first_trailing = trailing_tokens[0]
            if self.is_probable_unit(first_trailing):
                unit = first_trailing
                if len(trailing_tokens) > 1:
                    rest = self.clean_text(" ".join(trailing_tokens[1:]))
            else:
                rest = self.clean_text(" ".join(trailing_tokens))

        # Extra filtering: omschrijving moet voldoende tekst hebben
        if len(description) < 3:
            return None

        return {
            "description": description,
            "amount": amount_value,
            "unit": unit,
            "extra": rest,
            "raw_line": full_text,
        }

    def extract_from_pdf(self, pdf_path: Path) -> pd.DataFrame:
        rows = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(
                    keep_blank_chars=False,
                    use_text_flow=False
                )

                lines = self.group_words_into_lines(words)

                for line in lines:
                    parsed = self.parse_line(line)
                    if parsed is None:
                        continue

                    parsed["page"] = page_num
                    rows.append(parsed)

        return pd.DataFrame(rows)


if __name__ == "__main__":
    pdf_path = Path("../../../data/")
    extractor = OfferLineExtractor()
    df = extractor.extract_from_pdf(pdf_path)

    print(df.to_string(index=False))