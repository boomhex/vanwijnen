"""One-off backfill: fill Posten[].Pagina for extract.json files saved before
the Pagina feature existed, using the already-stored document.pdf and each
post's saved Brontekst/Omschrijving. Deterministic, no LLM calls.

Usage: env PYTHONPATH=. venv/bin/python scripts/backfill_pagina.py [storage_root]
"""

import json
import sys
from pathlib import Path

from services.extract_offer import find_post_page, read_pdf_with_pages


def backfill_file(extract_path: Path) -> int:
    pdf_path = extract_path.parent / 'document.pdf'
    if not pdf_path.exists():
        print(f'skip {extract_path} (no document.pdf)')
        return 0

    with open(extract_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    posten = data.get('Posten', [])
    missing = [p for p in posten if not str(p.get('Pagina') or '').strip() or p.get('Pagina') == 'ONBEKEND']
    if not missing:
        return 0

    _, page_texts = read_pdf_with_pages(str(pdf_path))
    changed = 0
    for post in missing:
        source_text = post.get('Brontekst') or post.get('Omschrijving')
        pagina = find_post_page(source_text, page_texts)
        if pagina != post.get('Pagina'):
            post['Pagina'] = pagina
            changed += 1

    if changed:
        with open(extract_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')

    return changed


def main() -> None:
    storage_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('storage')
    total_files = 0
    total_changed = 0
    for extract_path in sorted(storage_root.glob('*/*/extract.json')):
        changed = backfill_file(extract_path)
        if changed:
            total_files += 1
            total_changed += changed
            print(f'{extract_path}: {changed} posts updated')

    print(f'\nDone. {total_changed} posts updated across {total_files} files.')


if __name__ == '__main__':
    main()
