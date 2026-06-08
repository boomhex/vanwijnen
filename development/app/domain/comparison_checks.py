from domain.units import post_total_covers_mismatch, units_mismatch


def warning_for_offer(match_row: dict, offer: dict) -> str:
    warnings = []
    comparison_unit = match_row.get('Eenheid', '')
    matched_unit = (
        offer.get('Gematchte eenheid')
        or offer.get('Eenheid')
        or ''
    )

    if units_mismatch(comparison_unit, matched_unit) and not post_total_covers_mismatch(
        comparison_unit,
        matched_unit,
        offer,
    ):
        warnings.append(f'Eenheid wijkt af: vergelijking {comparison_unit}, offerte {matched_unit}.')

    score = str(offer.get('Overeenkomst', '')).strip()
    if score in {'1', '2'}:
        warnings.append(f'Lage overeenkomstscore ({score}). Controleer de match.')

    return ' '.join(warnings)
