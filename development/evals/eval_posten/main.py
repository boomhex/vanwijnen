from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
DEVELOPMENT_DIR = EVAL_DIR.parents[1]
APP_DIR = DEVELOPMENT_DIR / 'app'
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from domain.money import UNKNOWN, parse_decimal
from domain.units import normalize_unit


TESTCASES_DIR = EVAL_DIR / 'testcases'
RUNS_DIR = EVAL_DIR / 'runs'
SUMMARY_FIELDS = ['Naam aannemer', 'Totaalprijs inc. BTW', 'Totaalprijs exc. BTW']
POST_FIELDS = ['Omschrijving', 'Beschrijving', 'Categorie', 'Aantal', 'Eenheid', 'Eenheidsprijs', 'Totaalbedrag']
RICH_POST_FIELDS = [
    'PostType',
    'Status',
    'Code',
    'Regelnummer',
    'Subcategorie',
    'Werksoort',
    'Prijsbasis',
    'Inclusief',
    'Exclusief',
    'Voorwaarden',
    'DoorOpdrachtgever',
    'MatchHints',
    'Brontekst',
]


@dataclass(frozen=True)
class PostMatch:
    expected_index: int
    actual_index: int
    score: float


def normalize_text(value: Any) -> str:
    text = str(value or '').casefold()
    text = text.replace('m²', 'm2').replace('m¹', 'm1').replace('dznd', 'dzd')
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def known(value: Any) -> bool:
    if isinstance(value, list):
        return any(known(item) for item in value)
    text = str(value or '').strip()
    return bool(text) and text.upper() != UNKNOWN


def normalized_money(value: Any) -> str:
    decimal = parse_decimal(value)
    return str(decimal) if decimal is not None else normalize_text(value)


def values_equal(expected: Any, actual: Any, *, money: bool = False, unit: bool = False) -> bool:
    if money:
        return normalized_money(expected) == normalized_money(actual)
    if unit:
        return normalize_unit(expected) == normalize_unit(actual)
    return normalize_text(expected) == normalize_text(actual)


def fuzzy_ratio(first: Any, second: Any) -> float:
    first_text = normalize_text(first)
    second_text = normalize_text(second)
    if not first_text or not second_text:
        return 0.0
    return SequenceMatcher(None, first_text, second_text).ratio()


def tokens(value: Any) -> set[str]:
    return set(normalize_text(value).split())


def token_overlap(first: Any, second: Any) -> float:
    first_tokens = tokens(first)
    second_tokens = tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens)


def post_match_score(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    title = fuzzy_ratio(expected.get('Omschrijving'), actual.get('Omschrijving'))
    description = fuzzy_ratio(expected.get('Beschrijving'), actual.get('Beschrijving'))
    overlap = token_overlap(expected.get('Omschrijving'), actual_text(actual))
    score = (0.55 * title) + (0.2 * description) + (0.25 * overlap)
    if values_equal(expected.get('Eenheid'), actual.get('Eenheid'), unit=True):
        score += 0.08
    if values_equal(expected.get('Eenheidsprijs'), actual.get('Eenheidsprijs'), money=True):
        score += 0.08
    if values_equal(expected.get('Totaalbedrag'), actual.get('Totaalbedrag'), money=True):
        score += 0.08
    return min(score, 1.0)


def actual_text(post: dict[str, Any]) -> str:
    parts = [
        post.get('Omschrijving', ''),
        post.get('Beschrijving', ''),
        post.get('Categorie', ''),
        post.get('Subcategorie', ''),
        post.get('Werksoort', ''),
        post.get('Brontekst', ''),
        ' '.join(str(item) for item in post.get('MatchHints', []) if item),
    ]
    return ' '.join(str(part or '') for part in parts)


def greedy_post_matches(expected_posts: list[dict[str, Any]], actual_posts: list[dict[str, Any]], threshold: float) -> list[PostMatch]:
    pairs = []
    for expected_index, expected in enumerate(expected_posts):
        for actual_index, actual in enumerate(actual_posts):
            pairs.append(PostMatch(expected_index, actual_index, post_match_score(expected, actual)))

    pairs.sort(key=lambda match: match.score, reverse=True)
    matches = []
    used_expected = set()
    used_actual = set()
    for match in pairs:
        if match.score < threshold:
            break
        if match.expected_index in used_expected or match.actual_index in used_actual:
            continue
        matches.append(match)
        used_expected.add(match.expected_index)
        used_actual.add(match.actual_index)
    return matches


def score_summary(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    fields = {}
    correct = 0
    checked = 0
    for field in SUMMARY_FIELDS:
        expected_value = expected.get(field)
        if not known(expected_value):
            continue

        checked += 1
        money = 'prijs' in field.casefold()
        ok = values_equal(expected_value, actual.get(field), money=money)
        correct += int(ok)
        fields[field] = {
            'correct': ok,
            'expected': expected_value,
            'actual': actual.get(field, UNKNOWN),
        }

    return {
        'checked': checked,
        'correct': correct,
        'accuracy': correct / checked if checked else None,
        'fields': fields,
    }


def score_posts(expected: dict[str, Any], actual: dict[str, Any], threshold: float) -> dict[str, Any]:
    expected_posts = [post for post in expected.get('Posten', []) if isinstance(post, dict)]
    actual_posts = [post for post in actual.get('Posten', []) if isinstance(post, dict)]
    matches = greedy_post_matches(expected_posts, actual_posts, threshold)

    matched_expected = {match.expected_index for match in matches}
    matched_actual = {match.actual_index for match in matches}
    field_checked = 0
    field_correct = 0
    field_errors = []

    for match in matches:
        expected_post = expected_posts[match.expected_index]
        actual_post = actual_posts[match.actual_index]
        for field in POST_FIELDS:
            expected_value = expected_post.get(field)
            if not known(expected_value):
                continue

            field_checked += 1
            money = field in {'Aantal', 'Eenheidsprijs', 'Totaalbedrag'}
            unit = field == 'Eenheid'
            ok = values_equal(expected_value, actual_post.get(field), money=money, unit=unit)
            field_correct += int(ok)
            if not ok:
                field_errors.append({
                    'field': field,
                    'expected_post': expected_post.get('Omschrijving', ''),
                    'actual_post': actual_post.get('Omschrijving', ''),
                    'expected': expected_value,
                    'actual': actual_post.get(field, UNKNOWN),
                })

    missing = [
        expected_posts[index].get('Omschrijving', '')
        for index in range(len(expected_posts))
        if index not in matched_expected
    ]
    extra = [
        actual_posts[index].get('Omschrijving', '')
        for index in range(len(actual_posts))
        if index not in matched_actual
    ]

    return {
        'expected_posts': len(expected_posts),
        'actual_posts': len(actual_posts),
        'matched_posts': len(matches),
        'post_recall': len(matches) / len(expected_posts) if expected_posts else None,
        'post_precision': len(matches) / len(actual_posts) if actual_posts else None,
        'field_accuracy': field_correct / field_checked if field_checked else None,
        'field_checked': field_checked,
        'field_correct': field_correct,
        'missing': missing,
        'extra': extra,
        'field_errors': field_errors[:50],
        'matches': [
            {
                'score': round(match.score, 4),
                'expected': expected_posts[match.expected_index].get('Omschrijving', ''),
                'actual': actual_posts[match.actual_index].get('Omschrijving', ''),
            }
            for match in matches
        ],
    }


def score_rich_schema(actual: dict[str, Any]) -> dict[str, Any]:
    posts = [post for post in actual.get('Posten', []) if isinstance(post, dict)]
    if not posts:
        return {'posts': 0, 'coverage': None, 'fields': {}}

    fields = {}
    for field in RICH_POST_FIELDS:
        filled = sum(1 for post in posts if known(post.get(field)))
        fields[field] = {
            'filled': filled,
            'total': len(posts),
            'coverage': filled / len(posts),
        }

    filled_slots = sum(field_score['filled'] for field_score in fields.values())
    total_slots = len(posts) * len(RICH_POST_FIELDS)
    return {
        'posts': len(posts),
        'coverage': filled_slots / total_slots if total_slots else None,
        'fields': fields,
    }


def score_case(case: dict[str, Any], threshold: float) -> dict[str, Any]:
    actual_path = resolve_path(case.get('ExtractPath'))
    if actual_path is None:
        raise ValueError('Case is missing ExtractPath')

    expected = case.get('Expected')
    if not isinstance(expected, dict):
        raise ValueError('Case is missing Expected object')

    actual = load_json(actual_path)
    return {
        'actual_path': str(actual_path),
        'summary': score_summary(expected, actual),
        'posts': score_posts(expected, actual, threshold),
        'rich_schema': score_rich_schema(actual),
    }


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return data


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return path
    return DEVELOPMENT_DIR / path


def testcase_name_from_extract(path: Path) -> str:
    offer = path.parent.name
    project = path.parent.parent.name
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{project}__{offer}')


def generate_cases(extract_paths: list[Path], *, overwrite: bool) -> None:
    TESTCASES_DIR.mkdir(parents=True, exist_ok=True)
    for extract_path in extract_paths:
        extract_path = resolve_path(extract_path) or extract_path
        actual = load_json(extract_path)
        case = {
            'ExtractPath': str(extract_path.relative_to(DEVELOPMENT_DIR)),
            'Expected': {
                field: actual.get(field, UNKNOWN)
                for field in SUMMARY_FIELDS
            },
        }
        case['Expected']['Posten'] = actual.get('Posten', [])

        output_path = TESTCASES_DIR / f'{testcase_name_from_extract(extract_path)}.json'
        if output_path.exists() and not overwrite:
            raise FileExistsError(f'{output_path} already exists; use --overwrite')
        output_path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Wrote {output_path}')


def write_report(path: Path, scores: dict[str, Any]) -> None:
    lines = ['# Extraction Eval Report', '']
    total_expected = sum(score['posts']['expected_posts'] for score in scores.values())
    total_matched = sum(score['posts']['matched_posts'] for score in scores.values())
    total_actual = sum(score['posts']['actual_posts'] for score in scores.values())
    total_field_checked = sum(score['posts']['field_checked'] for score in scores.values())
    total_field_correct = sum(score['posts']['field_correct'] for score in scores.values())

    if total_expected:
        lines.extend([
            '## Overall',
            f'- Post recall: {total_matched / total_expected:.2%}',
            f'- Post precision: {total_matched / total_actual:.2%}' if total_actual else '- Post precision: n/a',
            f'- Field accuracy: {total_field_correct / total_field_checked:.2%}' if total_field_checked else '- Field accuracy: n/a',
            '',
        ])

    for case_name, score in scores.items():
        posts = score['posts']
        rich = score['rich_schema']
        recall = posts['post_recall']
        precision = posts['post_precision']
        field_accuracy = posts['field_accuracy']
        rich_coverage = rich['coverage']
        lines.extend([
            f'## {case_name}',
            f'- Post recall: {"n/a" if recall is None else f"{recall:.2%}"}',
            f'- Post precision: {"n/a" if precision is None else f"{precision:.2%}"}',
            f'- Field accuracy: {"n/a" if field_accuracy is None else f"{field_accuracy:.2%}"}',
            f'- Rich schema coverage: {"n/a" if rich_coverage is None else f"{rich_coverage:.2%}"}',
            f'- Missing posts: {len(posts["missing"])}',
            f'- Extra posts: {len(posts["extra"])}',
            '',
        ])

    path.write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate offer post extraction against golden snapshots.')
    parser.add_argument('--case', action='append', type=Path, help='Path to testcase JSON. Defaults to testcases/*.json.')
    parser.add_argument('--run-name', default=None, help='Name for the run directory.')
    parser.add_argument('--threshold', type=float, default=0.72, help='Fuzzy threshold for matching expected to actual posts.')
    parser.add_argument('--generate-from-extract', action='append', type=Path, help='Create golden testcase from extract.json.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite generated testcase files.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate_from_extract:
        generate_cases(args.generate_from_extract, overwrite=args.overwrite)
        return

    case_paths = args.case or sorted(TESTCASES_DIR.glob('*.json'))
    if not case_paths:
        raise SystemExit(f'No extraction testcases found in {TESTCASES_DIR}')

    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    scores = {}
    for path in case_paths:
        case = load_json(path)
        score = score_case(case, args.threshold)
        output_path = run_dir / f'{path.stem}.score.json'
        output_path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding='utf-8')
        scores[path.stem] = score

    write_report(run_dir / 'report.md', scores)
    print(f'Wrote extraction eval run to {run_dir}')


if __name__ == '__main__':
    main()
