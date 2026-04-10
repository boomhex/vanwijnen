import pandas as pd
from pathlib import Path
import os
from typing import List
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re

class LocateFile:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
    
    @staticmethod
    def find_file(folder: Path, file: Path) -> Path:
        for obj in os.listdir(folder):
            if str(file) == str(obj):
                return folder/file
            elif (folder/obj).is_dir():
                search = LocateFile.find_file(folder/obj, file)
                if search:
                    return search
                else:
                    continue

    def find_files(self, files: List[Path]) -> List[Path]:
        paths = []
        for file in files:
            fp = self.find_file(self.folder, file)
            paths.append(fp)
        return paths

class ExtractAmount:
    def __init__(self, model="textgain/allnli-GroNLP-bert-base-dutch-cased"):
        self.amount_pattern = re.compile(
            r"(€\s*)?\d{1,3}(?:\.\d{3})*,(?:\d{2}|=)|\b\d+(?:,\d{2}|,=)\b"
        )
        self.model = SentenceTransformer(model)

    def extract_first_amount(self, text: str) -> str | None:
        match = re.search(self.amount_pattern, text)
        return match.group(0) if match else None

    def make_windows(self, lines: list[str], window_sizes=(1, 2, 3)) -> list[dict]:
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

    def has_amount(self, text: str) -> bool:
        return re.search(self.amount_pattern, text) is not None

    def find_amounts(self, text: str) -> list[str]:
        return self.amount_pattern.findall(text)

    @staticmethod
    def clean_text(text: str):
        return ' '.join(text.lower().split())

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

    def pass_filter(self, line: str) -> bool:
        if not self.has_amount(line):
            return False
        return True

    def filter_lines(self, lines: List[str]) -> List[str]:
        filtered = []
        for window in lines:
            if self.pass_filter(window["text"]):
                filtered.append(window)
        return filtered

    def clean_amount(self, amount: str) -> float:
        cleaned = amount.replace("€", "").replace(".", "").replace(",", ".").replace("=", "").strip()
        return float(cleaned)

    def best_window(self, windows, sim_matrix):
        print(sim_matrix)

    def find_amount_query(
        self,
        query: str,
        file_path: Path
    ) -> float | None:
        lines = ExtractAmount.extract_lines(file_path)
        # for line in lines:
        #     print(line)
        windows = self.make_windows(lines)
        candidate_windows = self.filter_lines(windows)
        candidates = [window["text"] for window in candidate_windows]

        candidate_embeddings = self.model.encode(candidates, convert_to_tensor=True)
        query_embeddings = self.model.encode(query, convert_to_tensor=True)
        sim_matrix = util.cos_sim(query_embeddings, candidate_embeddings)

        # for score in zip(sim_matrix.tolist()[0], candidates):
        #     print(score)

        best_candidate = candidates[sim_matrix.argmax().item()]#self.best_window(windows, sim_matrix)["text"]
        amount = self.extract_first_amount(best_candidate)
        amount_numeric = self.clean_amount(amount)
        return amount_numeric

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype={"Totaalbedrag": float})
    return df

def main():
    data_path = Path("dataset.csv")
    df = load_data(data_path)

    files = list(df["Filename"])        # Find actual file paths
    locater = LocateFile(Path("../../data"))
    file_paths = locater.find_files(files)

    df["Filepath"] = file_paths

    extractor = ExtractAmount()

    found_amounts = [
        extractor.find_amount_query("totaalbedrag", fp)
        for fp in file_paths
    ]

    df["FoundAmounts"] = found_amounts
    print(df[["Filename", "Totaalbedrag", "FoundAmounts"]])


if __name__ == "__main__":
    main()
