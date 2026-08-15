from decimal import Decimal

from domain.money import parse_decimal
from domain.units import post_total_covers_mismatch, units_mismatch

QUANTITY_DIFFERENCE_RATIO = Decimal('0.05')
QUANTITY_DIFFERENCE_ABSOLUTE = Decimal('0.01')


def warnings_for_offer(match_row: dict, offer: dict) -> list[str]:
    warnings = []

    if offer.get('Ongekoppeld'):
        warnings.append(
            'Gematchte offertepost kon niet worden teruggevonden in de extractie; '
            'deze waarden zijn niet geverifieerd tegen de offerte zelf.'
        )

    # Get the unit matched by LLM
    comparison_unit = match_row.get('Eenheid', '')
    matched_unit = (
        offer.get('Gematchte eenheid')
        or offer.get('Eenheid')
        or ''
    )

    # Check if unit matches with the comparison
    if units_mismatch(comparison_unit, matched_unit) and not post_total_covers_mismatch(
        comparison_unit,
        matched_unit,
        offer,
    ):
        warnings.append(f'Eenheid wijkt af: vergelijking {comparison_unit}, offerte {matched_unit}.')

    # Warn for low comparison score
    score = str(offer.get('Overeenkomst', '')).strip()
    if score in {'1', '2'}:
        warnings.append(f'Lage overeenkomstscore ({score}). Controleer de match.')

    quantity_warning = quantity_difference_warning(match_row, offer)
    if quantity_warning:
        warnings.append(quantity_warning)

    return warnings


def warning_for_offer(match_row: dict, offer: dict) -> str:
    return ' '.join(warnings_for_offer(match_row, offer))


def quantity_difference_warning(match_row: dict, offer: dict) -> str:
    comparison_amount = parse_decimal(match_row.get('Aantal'))
    offer_amount = parse_decimal(offer.get('Aantal'))
    if comparison_amount is None or offer_amount is None:
        return ''

    comparison_unit = match_row.get('Eenheid', '')
    offer_unit = offer.get('Gematchte hoeveelheid eenheid') or offer.get('Gematchte eenheid') or offer.get('Eenheid') or ''
    if units_mismatch(comparison_unit, offer_unit):
        return ''

    difference = abs(comparison_amount - offer_amount)
    if difference <= QUANTITY_DIFFERENCE_ABSOLUTE:
        return ''

    baseline = abs(comparison_amount)
    if baseline == 0:
        if difference > QUANTITY_DIFFERENCE_ABSOLUTE:
            return f'Hoeveelheid wijkt af: vergelijking {match_row.get("Aantal")} {comparison_unit}, offerte {offer.get("Aantal")} {offer_unit}.'
        return ''

    if difference / baseline <= QUANTITY_DIFFERENCE_RATIO:
        return ''

    return f'Hoeveelheid wijkt af: vergelijking {match_row.get("Aantal")} {comparison_unit}, offerte {offer.get("Aantal")} {offer_unit}.'
