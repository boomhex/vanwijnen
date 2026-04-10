import pdfplumber
import pandas as pd
from pathlib import Path

pdf_path = Path("../../data/03.02_Offertes_ontvangen/10.31_totaal_sloopwerk/Bork/202500707-06-11-2025.pdf")

settings = {
    "vertical_strategy": "text",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3,
    "text_x_tolerance": 2,
    "text_y_tolerance": 2,
}

with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[2]

    tables = page.extract_tables(table_settings=settings)

    print(f"Aantal tabellen: {len(tables)}")

    for i, table in enumerate(tables, start=1):
        print(f"\n--- Tabel {i} ---")
        df = pd.DataFrame(table)
        print(df.head(20))