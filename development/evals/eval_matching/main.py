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

from domain.money import UNKNOWN, calculate_total, parse_decimal
from domain.units import normalize_unit
from matching.comparison_prompt import MATCH_RESPONSE_SCHEMA


DEFAULT_OFFER_ROOT = APP_DIR / 'storage' / 'test' / '22.31_baksteen'
DEFAULT_TESTCASES_DIR = EVAL_DIR / 'testcases'
RUNS_DIR = EVAL_DIR / 'runs'


SYNONYMS = {
    'gebogen': ['rond'],
    'rond': ['gebogen'],
    'wilverband': ['wildverband'],
    'wildverband': ['wilverband'],
    'vermetselen': ['metselen', 'gevelsteen'],
    'metselen': ['vermetselen', 'metselwerk'],
    'gevelsteen': ['wf', 'handvorm', 'vormbaksteen', 'baksteen'],
    'doorstrijken': ['doorgestreken', 'gepointerd', 'voegwerk'],
    'doorgestreken': ['doorstrijken', 'gepointerd'],
    'accentsteen': ['verdiept', 'zaagwerk', 'strooisteen'],
    'zagen': ['zaagwerk', 'zaagsnede'],
    'zaagwerk': ['zagen', 'gezaagd'],
    'steen': ['stenen', 'gevelsteen', 'baksteen'],
    'stenen': ['steen', 'gevelsteen', 'baksteen'],
    'dzd': ['dznd', 'duizend'],
    'dznd': ['dzd', 'duizend'],
    'st': ['stuk'],
    'stuk': ['st'],
    'mu': ['uur'],
    'uur': ['mu'],
    'verreiker': ['opperwerk', 'vooropperen'],
    'inzet': ['huur', 'materieel'],
}


STOPWORDS = {
    'aan',
    'de',
    'een',
    'en',
    'het',
    'in',
    'met',
    'of',
    'op',
    'per',
    'te',
    'ten',
    'ter',
    't.b.v',
    'van',
    'voor',
}


@dataclass(frozen=True)
class Candidate:
    score: float
    offer_name: str
    post: dict[str, Any]
    reasons: list[str]
    calculated_total: str


def normalize_text(value: Any) -> str:
    text = str(value or '').casefold()
    replacements = {
        'm²': 'm2',
        'm¹': 'm1',
        'dznd': 'dzd',
        'wilverband': 'wildverband',
        '€': ' eur ',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def text_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_text(value).split()
        if token and token not in STOPWORDS
    }


def expand_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(SYNONYMS.get(token, []))
    return expanded


def fuzzy_ratio(first: Any, second: Any) -> float:
    first_text = normalize_text(first)
    second_text = normalize_text(second)
    if not first_text or not second_text:
        return 0.0
    return SequenceMatcher(None, first_text, second_text).ratio()


def token_overlap(needle: Any, haystack: Any) -> float:
    needle_tokens = expand_tokens(text_tokens(needle))
    haystack_tokens = expand_tokens(text_tokens(haystack))
    if not needle_tokens or not haystack_tokens:
        return 0.0
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)


def list_text(value: Any) -> str:
    if isinstance(value, list):
        return ' '.join(str(item) for item in value)
    return str(value or '')


def offer_search_text(post: dict[str, Any]) -> str:
    fields = [
        'Omschrijving',
        'Beschrijving',
        'Categorie',
        'Subcategorie',
        'Werksoort',
        'Prijsbasis',
        'Brontekst',
        'PostType',
        'Status',
        'Code',
    ]
    parts = [str(post.get(field) or '') for field in fields]
    parts.extend([
        list_text(post.get('Inclusief')),
        list_text(post.get('Exclusief')),
        list_text(post.get('Voorwaarden')),
        list_text(post.get('DoorOpdrachtgever')),
        list_text(post.get('MatchHints')),
    ])
    return ' '.join(parts)


def infer_post_type(post: dict[str, Any]) -> str:
    text = normalize_text(' '.join([
        str(post.get('Omschrijving') or ''),
        str(post.get('Beschrijving') or ''),
    ]))
    if 'toeslag' in text or 'meerprijs' in text:
        return 'surcharge'
    if 'regie' in text or 'uur' in text or 'mu' in text:
        return 'regie'
    if 'alternatief' in text:
        return 'alternative'
    if 'optie' in text:
        return 'option'
    return ''


def matching_code(comparison_post: dict[str, Any], offer_post: dict[str, Any]) -> bool:
    comparison_code = normalize_text(comparison_post.get('Code'))
    offer_code = normalize_text(offer_post.get('Code'))
    return bool(comparison_code and offer_code and comparison_code == offer_code and comparison_code != normalize_text(UNKNOWN))


def unit_matches(comparison_post: dict[str, Any], offer_post: dict[str, Any]) -> bool:
    comparison_unit = normalize_unit(comparison_post.get('Eenheid'))
    offer_unit = normalize_unit(offer_post.get('Eenheid'))
    return bool(comparison_unit and offer_unit and comparison_unit == offer_unit)


def score_candidate(comparison_post: dict[str, Any], offer_name: str, offer_post: dict[str, Any]) -> Candidate:
    comparison_text = comparison_post.get('Omschrijving', '')
    offer_title = offer_post.get('Omschrijving', '')
    offer_description = offer_post.get('Beschrijving', '')
    offer_all = offer_search_text(offer_post)

    score = 0.0
    reasons = []

    title_score = fuzzy_ratio(comparison_text, offer_title)
    if title_score:
        score += 4.0 * title_score
        reasons.append(f'title={title_score:.2f}')

    description_score = fuzzy_ratio(comparison_text, offer_description)
    if description_score:
        score += 2.0 * description_score
        reasons.append(f'description={description_score:.2f}')

    overlap_score = token_overlap(comparison_text, offer_all)
    if overlap_score:
        score += 3.0 * overlap_score
        reasons.append(f'token_overlap={overlap_score:.2f}')

    hint_score = token_overlap(comparison_text, list_text(offer_post.get('MatchHints')))
    if hint_score:
        score += 1.5 * hint_score
        reasons.append(f'hints={hint_score:.2f}')

    if unit_matches(comparison_post, offer_post):
        score += 1.25
        reasons.append('unit_match')

    expected_type = infer_post_type(comparison_post)
    actual_type = str(offer_post.get('PostType') or '').strip()
    if expected_type and expected_type == actual_type:
        score += 1.5
        reasons.append(f'post_type={actual_type}')

    if matching_code(comparison_post, offer_post):
        score += 5.0
        reasons.append('code_match')

    calculated_total = calculate_candidate_total(comparison_post, offer_post)
    return Candidate(score, offer_name, offer_post, reasons, calculated_total)


def calculate_candidate_total(comparison_post: dict[str, Any], offer_post: dict[str, Any]) -> str:
    if parse_decimal(offer_post.get('Totaalbedrag')) is not None:
        return str(offer_post.get('Totaalbedrag'))
    return calculate_total(
        str(comparison_post.get('Aantal') or ''),
        str(offer_post.get('Eenheidsprijs') or ''),
        str(offer_post.get('Totaalbedrag') or UNKNOWN),
    )


def top_candidates_for_offer(
    comparison_post: dict[str, Any],
    offer_name: str,
    offer_posts: list[dict[str, Any]],
    *,
    limit: int,
) -> list[Candidate]:
    scored = [
        score_candidate(comparison_post, offer_name, offer_post)
        for offer_post in offer_posts
        if isinstance(offer_post, dict)
    ]
    scored.sort(key=lambda candidate: candidate.score, reverse=True)
    top = [candidate for candidate in scored if candidate.score > 0][:limit]

    if not any(unit_matches(comparison_post, candidate.post) for candidate in top):
        same_unit = [
            candidate
            for candidate in scored
            if unit_matches(comparison_post, candidate.post)
            and token_overlap(comparison_post.get('Omschrijving', ''), offer_search_text(candidate.post)) > 0
        ]
        if same_unit:
            best_same_unit = same_unit[0]
            if best_same_unit not in top:
                top = (top + [best_same_unit])[:limit]

    return top


def load_json_like(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads('{' + text.strip().strip(',') + '}')

    if not isinstance(data, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return data


def load_testcases(paths: list[Path]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    cases = []
    for path in paths:
        case = load_json_like(path)
        expected_path = path.with_suffix('.out')
        expected = load_json_like(expected_path) if expected_path.exists() else {}
        cases.append((path.stem, case, expected))
    return cases


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return path
    return DEVELOPMENT_DIR / path


def case_offer_root(case: dict[str, Any], cli_offer_root: Path | None) -> Path:
    return cli_offer_root or resolve_path(case.get('OfferRoot')) or DEFAULT_OFFER_ROOT


def load_offer_results(offer_root: Path) -> list[dict[str, Any]]:
    if offer_root.is_file():
        paths = [offer_root]
    else:
        paths = sorted(offer_root.glob('*/extract.json'))

    offer_results = []
    for path in paths:
        data = load_json_like(path)
        offer_results.append({
            'Bestand': f'{path.parent.name}.pdf',
            'ExtractPath': str(path),
            'Posten': data.get('Posten', []),
        })

    if not offer_results:
        raise ValueError(f'No offer extracts found in {offer_root}')
    return offer_results


def known_match_posts(offer: dict[str, Any]) -> list[str]:
    posts = offer.get('Gematchte posten') or []
    if isinstance(posts, str):
        posts = [posts]

    matched_description = offer.get('Gematchte omschrijving')
    if matched_description:
        posts = [*posts, matched_description]

    return [
        str(post)
        for post in posts
        if normalize_text(post) and normalize_text(post) != normalize_text(UNKNOWN)
        and not re.fullmatch(r'\d+\s+posten?', normalize_text(post))
    ]


def comparison_to_testcase(comparison: dict[str, Any], offer_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    case = {
        'OfferRoot': str(offer_root.relative_to(DEVELOPMENT_DIR)),
        'Posten': comparison.get('Posten', []),
    }
    expected_rows = []

    for row in comparison.get('MatchedPosten', []):
        offers = row.get('Offertes') or []
        if isinstance(offers, dict):
            offers = [
                {'Bestand': offer_name, **offer_data}
                for offer_name, offer_data in offers.items()
                if isinstance(offer_data, dict)
            ]

        expected_offers = []
        for offer in offers:
            posts = known_match_posts(offer)
            if posts:
                expected_offers.append({
                    'Bestand': offer.get('Bestand', ''),
                    'Gematchte posten': sorted(set(posts)),
                })

        if expected_offers:
            expected_rows.append({
                'Omschrijving': row.get('Omschrijving', ''),
                'Offertes': expected_offers,
            })

    expected = {'ExpectedMatches': expected_rows}
    return case, expected


def generate_testcases(comparison_paths: list[Path], *, overwrite: bool) -> None:
    for comparison_path in comparison_paths:
        comparison_path = resolve_path(comparison_path) or comparison_path
        comparison = load_json_like(comparison_path)
        offer_root = comparison_path.parent
        case, expected = comparison_to_testcase(comparison, offer_root)

        name = offer_root.name
        input_path = DEFAULT_TESTCASES_DIR / f'{name}.in'
        expected_path = DEFAULT_TESTCASES_DIR / f'{name}.out'
        if not overwrite and (input_path.exists() or expected_path.exists()):
            raise FileExistsError(f'{input_path} or {expected_path} already exists; use --overwrite')

        input_path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding='utf-8')
        expected_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Wrote {input_path} and {expected_path}')


def build_candidate_result(case: dict[str, Any], offer_results: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    rows = []
    for comparison_post in case.get('Posten', []):
        if not isinstance(comparison_post, dict):
            continue

        offers = []
        for offer_result in offer_results:
            offer_name = str(offer_result.get('Bestand') or '')
            candidates = top_candidates_for_offer(
                comparison_post,
                offer_name,
                offer_result.get('Posten', []),
                limit=limit,
            )
            offers.append({
                'Bestand': offer_name,
                'Candidates': [
                    candidate_to_json(candidate)
                    for candidate in candidates
                ],
            })

        rows.append({
            'Omschrijving': comparison_post.get('Omschrijving', ''),
            'Aantal': comparison_post.get('Aantal', ''),
            'Eenheid': comparison_post.get('Eenheid', ''),
            'Offertes': offers,
        })

    return {'CandidateMatches': rows}


def candidate_prompt_post(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        'Omschrijving': candidate.get('Omschrijving', ''),
        'Beschrijving': candidate.get('Beschrijving', ''),
        'Categorie': candidate.get('Categorie', ''),
        'Eenheid': candidate.get('Eenheid', ''),
        'Eenheidsprijs': candidate.get('Eenheidsprijs', ''),
        'Totaalbedrag': candidate.get('Totaalbedrag', ''),
        'CalculatedTotal': candidate.get('CalculatedTotal', ''),
        'PostType': candidate.get('PostType', ''),
        'Werksoort': candidate.get('Werksoort', ''),
        'Prijsbasis': candidate.get('Prijsbasis', ''),
        'MatchHints': candidate.get('MatchHints', []),
        'CandidateScore': candidate.get('Score', 0),
    }


def build_llm_match_prompt(candidate_result: dict[str, Any]) -> str:
    rows = []
    for row in candidate_result.get('CandidateMatches', []):
        offers = []
        for offer in row.get('Offertes', []):
            offers.append({
                'Bestand': offer.get('Bestand', ''),
                'Kandidaten': [
                    candidate_prompt_post(candidate)
                    for candidate in offer.get('Candidates', [])
                ],
            })

        rows.append({
            'Omschrijving': row.get('Omschrijving', ''),
            'Aantal': row.get('Aantal', ''),
            'Eenheid': row.get('Eenheid', ''),
            'Offertes': offers,
        })

    return '\n'.join([
        'Je koppelt begrotings-/vergelijkingsregels aan offerteposten.',
        'Je krijgt per vergelijkingsregel per offertebestand alleen voorgeselecteerde kandidaatposten.',
        'Kies per offertebestand de beste inhoudelijke match uit de kandidaten.',
        'Gebruik "single" als één kandidaat past en "group" als meerdere kandidaten samen nodig zijn.',
        'Gebruik "ONBEKEND" en een lege "Gematchte posten" lijst als geen kandidaat inhoudelijk goed past.',
        'Match nooit alleen op prijs, hoeveelheid of eenheid; de werkzaamheden moeten inhoudelijk overeenkomen.',
        'Bij PostType "unit_rate" mag je de hoeveelheid uit de vergelijkingsregel combineren met de eenheidsprijs van de offertepost als de eenheid inhoudelijk overeenkomt.',
        'Kopieer gematchte omschrijvingen letterlijk uit kandidaatveld "Omschrijving".',
        'Reageer ALLEEN met geldige JSON in dit formaat:',
        json.dumps({
            'MatchedPosten': [
                {
                    'Omschrijving': '...',
                    'Offertes': [
                        {
                            'Bestand': 'offerte-bestandsnaam.pdf',
                            'Match type': 'single',
                            'Gematchte omschrijving': '...',
                            'Gematchte posten': ['...'],
                            'Overeenkomst': '1-3',
                        },
                    ],
                },
            ],
        }, ensure_ascii=False, indent=2),
        'INPUT:',
        json.dumps(rows, ensure_ascii=False, indent=2),
    ])


def run_llm_match(candidate_result: dict[str, Any], *, model: str) -> tuple[dict[str, Any], str]:
    from services.extract_offer import ask_llm, parse_json_response

    prompt = build_llm_match_prompt(candidate_result)
    answer = ask_llm(
        prompt,
        response_schema=MATCH_RESPONSE_SCHEMA,
        model=model,
        label='matching_eval_final',
    )
    return parse_json_response(answer), answer


def candidate_to_json(candidate: Candidate) -> dict[str, Any]:
    return {
        'Score': round(candidate.score, 4),
        'Omschrijving': candidate.post.get('Omschrijving', ''),
        'Beschrijving': candidate.post.get('Beschrijving', ''),
        'Categorie': candidate.post.get('Categorie', ''),
        'Eenheid': candidate.post.get('Eenheid', ''),
        'Eenheidsprijs': candidate.post.get('Eenheidsprijs', ''),
        'Totaalbedrag': candidate.post.get('Totaalbedrag', ''),
        'CalculatedTotal': candidate.calculated_total,
        'PostType': candidate.post.get('PostType', ''),
        'Werksoort': candidate.post.get('Werksoort', ''),
        'Prijsbasis': candidate.post.get('Prijsbasis', ''),
        'MatchHints': candidate.post.get('MatchHints', []),
        'Reasons': candidate.reasons,
    }


def expected_matches(expected: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    rows = expected.get('MatchedPosten') or expected.get('ExpectedMatches') or []
    matches = {}
    for row in rows:
        comparison_description = str(row.get('Omschrijving') or '')
        offers = row.get('Offertes') or []
        if isinstance(offers, dict):
            offers = [
                {'Bestand': offer_name, **offer_data}
                for offer_name, offer_data in offers.items()
                if isinstance(offer_data, dict)
            ]
        for offer in offers:
            offer_name = str(offer.get('Bestand') or '')
            posts = known_match_posts(offer)
            matches[(comparison_description, offer_name)] = {
                normalize_text(post)
                for post in posts
                if normalize_text(post) and normalize_text(post) != normalize_text(UNKNOWN)
            }
    return matches


def score_candidate_result(result: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    expected_by_key = expected_matches(expected)
    if not expected_by_key:
        return {
            'has_expected': False,
            'candidate_recall_at_8': None,
            'checked_cells': 0,
            'found_cells': 0,
            'misses': [],
        }

    checked = 0
    found = 0
    misses = []
    for row in result.get('CandidateMatches', []):
        comparison_description = str(row.get('Omschrijving') or '')
        for offer in row.get('Offertes', []):
            offer_name = str(offer.get('Bestand') or '')
            expected_posts = expected_by_key.get((comparison_description, offer_name))
            if not expected_posts:
                continue

            checked += 1
            candidate_posts = {
                normalize_text(candidate.get('Omschrijving'))
                for candidate in offer.get('Candidates', [])
            }
            if expected_posts & candidate_posts:
                found += 1
            else:
                misses.append({
                    'Omschrijving': comparison_description,
                    'Bestand': offer_name,
                    'Expected': sorted(expected_posts),
                    'Candidates': sorted(candidate_posts),
                })

    return {
        'has_expected': True,
        'candidate_recall_at_8': found / checked if checked else None,
        'checked_cells': checked,
        'found_cells': found,
        'misses': misses,
    }


def result_offer_matches(result: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    matches = {}
    for row in result.get('MatchedPosten', []):
        comparison_description = str(row.get('Omschrijving') or '')
        offers = row.get('Offertes') or []
        if isinstance(offers, dict):
            offers = [
                {'Bestand': offer_name, **offer_data}
                for offer_name, offer_data in offers.items()
                if isinstance(offer_data, dict)
            ]

        for offer in offers:
            offer_name = str(offer.get('Bestand') or '')
            matches[(comparison_description, offer_name)] = {
                normalize_text(post)
                for post in known_match_posts(offer)
            }
    return matches


def score_llm_result(result: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    expected_by_key = expected_matches(expected)
    if not expected_by_key:
        return {
            'has_expected': False,
            'llm_match_accuracy': None,
            'checked_cells': 0,
            'correct_cells': 0,
            'misses': [],
        }

    actual_by_key = result_offer_matches(result)
    checked = 0
    correct = 0
    misses = []
    for key, expected_posts in expected_by_key.items():
        if not expected_posts:
            continue

        checked += 1
        actual_posts = actual_by_key.get(key, set())
        if expected_posts & actual_posts:
            correct += 1
        else:
            misses.append({
                'Omschrijving': key[0],
                'Bestand': key[1],
                'Expected': sorted(expected_posts),
                'Actual': sorted(actual_posts),
            })

    return {
        'has_expected': True,
        'llm_match_accuracy': correct / checked if checked else None,
        'checked_cells': checked,
        'correct_cells': correct,
        'misses': misses,
    }


def write_report(path: Path, all_scores: dict[str, Any], result_paths: dict[str, Path]) -> None:
    lines = ['# Matching Eval Report', '']
    total_checked = sum(score.get('checked_cells', 0) for score in all_scores.values() if score.get('has_expected'))
    total_found = sum(score.get('found_cells', 0) for score in all_scores.values() if score.get('has_expected'))
    llm_scores = [score.get('llm') for score in all_scores.values() if isinstance(score.get('llm'), dict)]
    llm_checked = sum(score.get('checked_cells', 0) for score in llm_scores if score.get('has_expected'))
    llm_correct = sum(score.get('correct_cells', 0) for score in llm_scores if score.get('has_expected'))
    if total_checked:
        lines.extend([
            '## Overall',
            f'- Candidate recall@8: {total_found / total_checked:.2%}',
            f'- Found cells: {total_found}/{total_checked}',
            f'- LLM match accuracy: {llm_correct / llm_checked:.2%}' if llm_checked else '- LLM match accuracy: not run',
            f'- LLM correct cells: {llm_correct}/{llm_checked}' if llm_checked else '- LLM correct cells: not run',
            '',
        ])

    for case_name, score in all_scores.items():
        lines.append(f'## {case_name}')
        if score.get('error'):
            lines.append(f'- Error: {score["error"]}')
            lines.append('')
            continue

        lines.append(f'- Candidates: `{result_paths[case_name].name}`')
        if not score.get('has_expected'):
            lines.append('- No expected matches found; candidate recall was not scored.')
        else:
            recall = score.get('candidate_recall_at_8')
            recall_text = 'n/a' if recall is None else f'{recall:.2%}'
            lines.append(f'- Candidate recall@8: {recall_text}')
            lines.append(f'- Found cells: {score["found_cells"]}/{score["checked_cells"]}')
            lines.append(f'- Misses: {len(score["misses"])}')
            llm_score = score.get('llm')
            if isinstance(llm_score, dict):
                accuracy = llm_score.get('llm_match_accuracy')
                accuracy_text = 'n/a' if accuracy is None else f'{accuracy:.2%}'
                lines.append(f'- LLM match accuracy: {accuracy_text}')
                lines.append(f'- LLM correct cells: {llm_score["correct_cells"]}/{llm_score["checked_cells"]}')
                lines.append(f'- LLM misses: {len(llm_score["misses"])}')
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate matching candidate retrieval.')
    parser.add_argument(
        '--case',
        action='append',
        type=Path,
        help='Path to a testcase .in file. Defaults to all testcases/*.in.',
    )
    parser.add_argument(
        '--offers',
        type=Path,
        default=None,
        help='Offer extract.json file or directory containing offer folders with extract.json. Overrides OfferRoot in testcases.',
    )
    parser.add_argument('--limit', type=int, default=8, help='Number of candidates per comparison row per offer.')
    parser.add_argument('--run-name', default=None, help='Name for the run directory.')
    parser.add_argument('--with-llm', action='store_true', help='Ask the LLM to choose final matches from candidates.')
    parser.add_argument('--llm-model', default='gemini-2.5-flash', help='Model to use with --with-llm.')
    parser.add_argument(
        '--generate-from-comparison',
        action='append',
        type=Path,
        help='Create .in/.out fixtures from an existing comparison.json.',
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite generated testcase files.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate_from_comparison:
        generate_testcases(args.generate_from_comparison, overwrite=args.overwrite)
        return

    case_paths = args.case or sorted(DEFAULT_TESTCASES_DIR.glob('*.in'))
    if not case_paths:
        raise SystemExit(f'No testcases found in {DEFAULT_TESTCASES_DIR}')

    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    scores = {}
    result_paths = {}

    for case_name, case, expected in load_testcases(case_paths):
        score_path = run_dir / f'{case_name}.score.json'
        try:
            offer_results = load_offer_results(case_offer_root(case, args.offers))
            result = build_candidate_result(case, offer_results, limit=args.limit)
            score = score_candidate_result(result, expected)

            result_path = run_dir / f'{case_name}.candidates.json'
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            result_paths[case_name] = result_path

            if args.with_llm:
                llm_result, llm_answer = run_llm_match(result, model=args.llm_model)
                llm_result_path = run_dir / f'{case_name}.llm_matches.json'
                llm_answer_path = run_dir / f'{case_name}.llm_response.txt'
                llm_score_path = run_dir / f'{case_name}.llm_score.json'
                llm_result_path.write_text(json.dumps(llm_result, ensure_ascii=False, indent=2), encoding='utf-8')
                llm_answer_path.write_text(llm_answer, encoding='utf-8')
                llm_score = score_llm_result(llm_result, expected)
                llm_score_path.write_text(json.dumps(llm_score, ensure_ascii=False, indent=2), encoding='utf-8')
                score['llm'] = llm_score
        except Exception as error:
            score = {
                'error': str(error),
                'has_expected': False,
                'candidate_recall_at_8': None,
                'checked_cells': 0,
                'found_cells': 0,
                'misses': [],
            }

        scores[case_name] = score
        score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding='utf-8')

    write_report(run_dir / 'report.md', scores, result_paths)
    print(f'Wrote matching eval run to {run_dir}')


if __name__ == '__main__':
    main()
