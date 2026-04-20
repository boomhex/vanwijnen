import pdfplumber
from pathlib import Path

class FolderCrawl:
    def __call__(self, path: Path) -> list[Path]:
        return self.crawl(path)

    def crawl(self, path: Path) -> list[Path]:
        return list(path.rglob("*.pdf"))

def extract_text(file):
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text())
    try:    
        return '\n'.join(text)
    except:
        return ''

def word_in_text(word, text):
    return word in text

def has_totaalprijs(text):
    words = [
        "totaalprijs",
        "aanneemsom",
        "totaal"
    ]

    if word_in_text("offerte", text):
        for word in words:
            if word_in_text(word, text):
                return True
    return False

def is_offerte(text):
    return word_in_text("offerte", text)

def main():
    crawler = FolderCrawl()
    files = crawler(Path("../../../data"))

    totaalprijs = 0
    offerte = 0
    for i, file in enumerate(files):
        if i % 10 == 0:
            print(f"File {i}/{len(files)}")
        text = extract_text(file).lower()
        if has_totaalprijs(text):
            totaalprijs += 1
        if is_offerte(text):
            offerte += 1

    print("Files checked: ", len(files))
    print("Offerte: ", offerte)
    print("Has totaalprijs: ", totaalprijs)

if __name__ == "__main__":
    main()
