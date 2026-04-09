from pdfminer.high_level import extract_text
from pathlib import Path

data_path = Path("../../data/03.02_Offertes_ontvangen/10.31_totaal_sloopwerk/Bork/202500707-06-11-2025.pdf")

text = extract_text(data_path)
print(text)
