OFFER_FIELDS: list[str] = ['Omschrijving', 'Beschrijving', 'Categorie', 'Aantal', 'Eenheid', 'Eenheidsprijs', 'Totaalbedrag']
COMPARISON_FIELDS: list[str] = ['Omschrijving', 'Aantal', 'Eenheid']

FIELD_TO_ATTR: dict[str, str] = {
    'Omschrijving': 'omschrijving',
    'Beschrijving': 'beschrijving',
    'Categorie': 'categorie',
    'Aantal': 'aantal',
    'Eenheid': 'eenheid',
    'Eenheidsprijs': 'eenheidsprijs',
    'Totaalbedrag': 'totaalbedrag',
}
