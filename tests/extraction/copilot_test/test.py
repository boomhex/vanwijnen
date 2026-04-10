import pdfplumber
import pandas as pd
from pathlib import Path

pdf_path = Path("../../../data/03.02_Offertes_ontvangen/12.00_grondwerk/Lolkema/12-11-2025 Openbegroting INFRA.pdf")

with pdfplumber.open(pdf_path) as pdf:

    text = []
    for page in pdf.pages:
        text.append(page.extract_text())
    text = "\n".join(text)

with open("new.txt", 'w') as f:
    f.write(text)
