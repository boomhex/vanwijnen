import pdfplumber
from pathlib import Path
import pandas as pd

PDF_PATH = Path(
    "../../../data/03.02_Offertes_ontvangen/22.31_baksteen_met_mortel/Postma/Nieuwbouw IKC Sint Nicolaasga.pdf"
)
DATASET_PATH = "post_dataset.csv"

def get_lines(path: Path) -> list[str]:
    text = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages):
            for line in page.extract_text().split('\n'):
                text.append([line, page_no])

    return text

def annotate_row(row):
    print(row)
    new_row = {
        "text" : row,
        "post" : input("Is dit een post (y/n) ") == 'y',
        "has_amount" : input("Bevat deze regel een bedrag? (y/n) ") == 'y',
        "has_unit" : input("Bevat deze regel een eenheid? (y/n) ") == 'y'
    }

    return new_row

def main():
    lines = get_lines(PDF_PATH)
    # print(lines)
    # exit()
    df = pd.read_csv(DATASET_PATH)

    for line, page in lines:
        row = annotate_row(line)
        row["page"] = page
        df.loc[len(df)] = row
    
    df.to_csv(
        DATASET_PATH.split('.')[0] + "(1)" + ".csv"
    )

if __name__ == "__main__":
    main()
