'''
This file is for development of extracting offers from pdfs.
'''

from gemini import ask_llm
from pathlib import Path
from argparse import ArgumentParser
import pdfplumber


def text_from_pdf(file: Path) -> str:
    text = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

    return "\n".join(text)


def read_prompt_text(fp: Path):
    with open(fp, 'r') as file:
        text = file.read()
    return text


def main(input_path: Path, output_folder: Path) -> None:
    # load file
    file_text = text_from_pdf(input_path)

    # extract output
    prompt = read_prompt_text("prompts/1.txt")
    extracted_text = ask_llm(prompt + file_text)

    # save output
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "out.txt", 'a') as file:
        file.write(extracted_text)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inputFile", type=Path, required=True)
    parser.add_argument("--outputFolder", type=Path, required=True)
    args = parser.parse_args()
    main(args.inputFile, args.outputFolder)
