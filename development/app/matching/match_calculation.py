from domain.money import calculate_total, first_known_value
from domain.units import normalize_unit
from matching.match_fields import match_type_for_offer


def calculate_offer_total(amount: str, offer: dict, comparison_unit: str | None = None) -> str:
    units = [
        comparison_unit,
        offer.get('Gematchte eenheid'),
        offer.get('Eenheid'),
    ]
    unit_price = offer.get('Eenheidsprijs')
    fallback_total = offer.get('Totaalbedrag')

    if match_type_for_offer(offer) == 'group':
        return first_known_value(fallback_total, unit_price)

    if any('post' in normalize_unit(unit) for unit in units):
        return first_known_value(fallback_total, unit_price)

    return calculate_total(amount, unit_price, fallback_total)
