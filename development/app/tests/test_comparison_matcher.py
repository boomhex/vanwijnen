from decimal import Decimal

from services.comparison_matcher import ComparisonMatcher


def test_parse_decimal():
    assert ComparisonMatcher.parse_decimal('1.234,56') == Decimal('1234.56')
    assert ComparisonMatcher.parse_decimal('1234.56') == Decimal('1234.56')
    assert ComparisonMatcher.parse_decimal('ONBEKEND') is None
    assert ComparisonMatcher.parse_decimal('') is None


def test_calculate_total():
    # amount * unit_price
    assert ComparisonMatcher.calculate_total('2', '3.50') == '7.00'
    # missing values returns fallback
    assert ComparisonMatcher.calculate_total('ONBEKEND', '3.50', '42') == '42'
    # decimals with commas
    assert ComparisonMatcher.calculate_total('2,5', '2') == '5.00'


def test_recalculate_matched_posts():
    comparison = {
        'MatchedPosten': [
            {
                'Omschrijving': 'Metselwerk',
                'Aantal': '3',
                'Eenheid': 'm2',
                'Offertes': {
                    'postma.pdf': {
                        'Gematchte eenheid': 'm2',
                        'Eenheidsprijs': '12,50',
                        'Totaalbedrag': '1',
                    },
                    'zuidema.pdf': {
                        'Gematchte eenheid': 'post',
                        'Eenheidsprijs': '900',
                        'Totaalbedrag': '2700',
                    },
                },
            },
        ],
    }

    matcher = ComparisonMatcher(folder_handler=None)

    assert matcher.recalculate_matched_posts(comparison) == [
        {
            'Omschrijving': 'Metselwerk',
            'Aantal': '3',
            'Eenheid': 'm2',
            'Offertes': {
                'postma.pdf': {
                    'Gematchte eenheid': 'm2',
                    'Eenheidsprijs': '12,50',
                    'Totaalbedrag': '37.50',
                },
                'zuidema.pdf': {
                    'Gematchte eenheid': 'post',
                    'Eenheidsprijs': '900',
                    'Totaalbedrag': '2700',
                },
            },
        },
    ]


def test_recalculate_matched_posts_refreshes_from_extract():
    comparison = {
        'MatchedPosten': [
            {
                'Omschrijving': 'Bouwplaatsinrichting',
                'Aantal': '3',
                'Eenheid': 'post',
                'Offertes': {
                    'lolkema.pdf': {
                        'Gematchte omschrijving': 'Bouwplaatsinrichting',
                        'Gematchte eenheid': 'post',
                        'Eenheidsprijs': '900',
                        'Totaalbedrag': '2700',
                    },
                },
            },
        ],
    }
    offer_results = [
        {
            'Bestand': 'lolkema.pdf',
            'Posten': [
                {
                    'Omschrijving': 'Bouwplaatsinrichting',
                    'Categorie': 'Bouwplaatskosten',
                    'Eenheid': 'm1',
                    'Eenheidsprijs': '900',
                    'Totaalbedrag': '3000',
                },
            ],
        },
    ]
    matcher = ComparisonMatcher(folder_handler=None)
    matcher.project_offer_results = lambda _project: offer_results

    result = matcher.recalculate_matched_posts(comparison, project=object())

    assert result[0]['Offertes']['lolkema.pdf']['Totaalbedrag'] == '3000'
    assert result[0]['Offertes']['lolkema.pdf']['Gematchte categorie'] == 'Bouwplaatskosten'


def test_recalculate_matched_posts_sums_group_matches_from_extract():
    comparison = {
        'MatchedPosten': [
            {
                'Omschrijving': 'Grondwerk bouwkundig',
                'Aantal': '1',
                'Eenheid': 'post',
                'Offertes': {
                    'lolkema.pdf': {
                        'Match type': 'group',
                        'Gematchte posten': [
                            'Ontgraven bouwkuip',
                            'Afwerken bouwkuip',
                        ],
                        'Totaalbedrag': 'ONBEKEND',
                    },
                },
            },
        ],
    }
    offer_results = [
        {
            'Bestand': 'lolkema.pdf',
            'Posten': [
                {
                    'Omschrijving': 'Ontgraven bouwkuip',
                    'Categorie': 'Grondwerk bouwkuip',
                    'Eenheid': 'm2',
                    'Eenheidsprijs': '1.50',
                    'Totaalbedrag': '1500.00',
                },
                {
                    'Omschrijving': 'Afwerken bouwkuip',
                    'Categorie': 'Grondwerk bouwkuip',
                    'Eenheid': 'm2',
                    'Eenheidsprijs': '2.50',
                    'Totaalbedrag': '2500.00',
                },
            ],
        },
    ]
    matcher = ComparisonMatcher(folder_handler=None)
    matcher.project_offer_results = lambda _project: offer_results

    result = matcher.recalculate_matched_posts(comparison, project=object())
    offer = result[0]['Offertes']['lolkema.pdf']

    assert offer['Match type'] == 'group'
    assert offer['Gematchte omschrijving'] == '2 posten'
    assert offer['Gematchte categorie'] == 'Grondwerk bouwkuip'
    assert offer['Gematchte categorieen'] == ['Grondwerk bouwkuip']
    assert offer['Totaalbedrag'] == '4000.00'


def test_calculate_offer_total_uses_total_when_comparison_unit_is_post():
    assert ComparisonMatcher.calculate_offer_total(
        '1',
        {
            'Gematchte eenheid': 'm1',
            'Eenheidsprijs': '12.00',
            'Totaalbedrag': '4800.00',
        },
        comparison_unit='post',
    ) == '4800.00'


def test_units_mismatch_normalizes_common_unit_spellings():
    assert ComparisonMatcher.units_mismatch('m2', 'm²') is False
    assert ComparisonMatcher.units_mismatch('m1', 'm¹') is False
    assert ComparisonMatcher.units_mismatch('dzd', 'st') is True
    assert ComparisonMatcher.units_mismatch('dzd', 'ONBEKEND') is False


def test_warning_for_offer():
    warning = ComparisonMatcher.warning_for_offer(
        {'Eenheid': 'dzd'},
        {'Gematchte eenheid': 'st', 'Overeenkomst': '1'},
    )

    assert 'Eenheid wijkt af' in warning
    assert 'Lage overeenkomstscore' in warning

    warning = ComparisonMatcher.warning_for_offer(
        {'Eenheid': 'dzd'},
        {'Gematchte eenheid': 'dzd', 'Overeenkomst': '2'},
    )

    assert 'Lage overeenkomstscore (2)' in warning


def test_warning_for_offer_allows_post_mismatch_with_total():
    warning = ComparisonMatcher.warning_for_offer(
        {'Eenheid': 'post'},
        {
            'Gematchte eenheid': 'm1',
            'Totaalbedrag': '4800.00',
            'Overeenkomst': '3',
        },
    )

    assert 'Eenheid wijkt af' not in warning


def test_warning_for_offer_keeps_post_mismatch_without_total():
    warning = ComparisonMatcher.warning_for_offer(
        {'Eenheid': 'post'},
        {
            'Gematchte eenheid': 'm1',
            'Totaalbedrag': 'ONBEKEND',
            'Overeenkomst': '3',
        },
    )

    assert 'Eenheid wijkt af' in warning
