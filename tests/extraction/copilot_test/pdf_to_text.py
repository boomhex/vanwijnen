import pdfplumber
import pandas as pd
from pathlib import Path

pdf_path = Path(
    "../../../data/03.02_Offertes_ontvangen/42.40.a_gietvloeren_vloercoatings/Facadec/Offerte FBV - Nieuwbouw IKC St Nicolaasga.pdf"
)

with pdfplumber.open(pdf_path) as pdf:

    text = []
    for page in pdf.pages:
        text.append(page.extract_text())
    text = "\n".join(text)

filename = pdf_path.name.split('.')[0] + '.txt'
filename = '_'.join(filename.split(' '))

with open(filename, 'w') as f:
    f.write(text)
