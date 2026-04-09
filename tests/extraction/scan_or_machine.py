import pdfplumber
import os
from pathlib import Path

def detect_pdf_type(path):
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]

        words = page.extract_words()
        images = page.images

        if len(words) > 20:
            return "machine"
        elif len(images) > 0:
            return "scan"
        else:
            return "unknown"

def is_pdf(path: Path):
    root, ext = os.path.splitext(path)
    return ext == ".pdf"

def pdfs_in_folder(folder: Path):
    pdfs = []
    if folder.is_dir():
        for path in os.listdir(folder):
            if (folder/path).is_dir():
                pdfs.extend(pdfs_in_folder(folder/path))
            elif is_pdf(path):
                pdfs.append(folder/path)
    return pdfs

path = Path("../../data")
pdfs = pdfs_in_folder(path)
types = [detect_pdf_type(pdf) for pdf in pdfs]
machine = sum([int(t == "machine") for t in types])
scan = sum([int(t == "scan") for t in types])
unknown = sum([int(t == "unknown") for t in types])

print("Number of pdfs:", len(pdfs))
print("Machine %:", machine * 100 / (machine + scan + unknown), f"({machine})")
print("Scan %:", scan * 100 / (machine + scan + unknown), f"({scan})")
print("Unknown %:", unknown * 100 / (machine + scan + unknown))
