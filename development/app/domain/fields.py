OFFER_FIELDS: list[str] = ['Omschrijving', 'Categorie', 'Aantal', 'Eenheid', 'Eenheidsprijs', 'Totaalbedrag']
COMPARISON_FIELDS: list[str] = ['Omschrijving', 'Aantal', 'Eenheid']

FIELD_TO_ATTR: dict[str, str] = {
    'Omschrijving': 'omschrijving',
    'Categorie': 'categorie',
    'Aantal': 'aantal',
    'Eenheid': 'eenheid',
    'Eenheidsprijs': 'eenheidsprijs',
    'Totaalbedrag': 'totaalbedrag',
}
