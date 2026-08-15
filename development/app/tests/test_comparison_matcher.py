from decimal import Decimal

from domain.comparison_checks import warning_for_offer, warnings_for_offer
from domain.money import calculate_total, calculate_unit_price, parse_decimal
from domain.units import units_mismatch
from application.comparison_service import ComparisonService
from interface.right_side.comparison_page import MatchedPostenTable
from matching.comparison_prompt import match_prompt_offer_results
from matching.match_calculation import calculate_offer_total
from matching.match_lookup import find_extracted_post_by_description, find_extracted_posts_for_match
from matching.match_normalizer import complete_offer_info, complete_response, normalize_matched_posts
from services.comparison_matcher import ComparisonMatcher


def test_parse_decimal():
    assert parse_decimal('1.234,56') == Decimal('1234.56')
    assert parse_decimal('1234.56') == Decimal('1234.56')
    assert parse_decimal('ONBEKEND') is None
    assert parse_decimal('') is None


def test_calculate_total():
    # amount * unit_price
    assert calculate_total('2', '3.50') == '7.00'
    # missing values returns fallback
    assert calculate_total('ONBEKEND', '3.50', '42') == '42'
    # decimals with commas
    assert calculate_total('2,5', '2') == '5.00'


def test_calculate_unit_price_from_total_and_amount():
    assert calculate_unit_price('1200', '3') == '400.00'
    assert calculate_unit_price('1200', '3', '250') == '250'
    assert calculate_unit_price('1200', '0') == 'ONBEKEND'


def test_match_prompt_offer_results_keeps_only_matching_context_fields():
    assert match_prompt_offer_results([
        {
            'Bestand': 'offer.pdf',
            'Posten': [
                {
                    'Omschrijving': 'Post A',
                    'Beschrijving': 'Uitgebreide toelichting',
                    'Categorie': 'Cat',
                    'Aantal': '10',
                    'Eenheid': 'm2',
                    'Eenheidsprijs': '20',
                    'Totaalbedrag': '200',
                },
            ],
        },
    ]) == [
        {
            'Bestand': 'offer.pdf',
            'Posten': [
                {
                    'Omschrijving': 'Post A',
                    'Beschrijving': 'Uitgebreide toelichting',
                    'Categorie': 'Cat',
                    'Aantal': '10',
                    'Eenheid': 'm2',
                },
            ],
        },
    ]


def test_match_prompt_offer_results_includes_rich_fields_when_known():
    assert match_prompt_offer_results([
        {
            'Bestand': 'offer.pdf',
            'Posten': [
                {
                    'Omschrijving': 'Post A',
                    'Beschrijving': 'Uitgebreide toelichting',
                    'Categorie': 'Cat',
                    'Aantal': '10',
                    'Eenheid': 'm2',
                    'Eenheidsprijs': '20',
                    'Totaalbedrag': '200',
                    'Code': '44.31.10-a',
                    'Regelnummer': 'ONBEKEND',
                    'PostType': 'unit_rate',
                    'Status': 'included',
                    'Subcategorie': 'ONBEKEND',
                    'Werksoort': 'metselwerk',
                    'Prijsbasis': 'per m2',
                    'MatchHints': ['baksteen', 'wildverband', ''],
                    'Inclusief': ['iets'],
                },
            ],
        },
    ]) == [
        {
            'Bestand': 'offer.pdf',
            'Posten': [
                {
                    'Omschrijving': 'Post A',
                    'Beschrijving': 'Uitgebreide toelichting',
                    'Categorie': 'Cat',
                    'Aantal': '10',
                    'Eenheid': 'm2',
                    'Code': '44.31.10-a',
                    'PostType': 'unit_rate',
                    'Status': 'included',
                    'Werksoort': 'metselwerk',
                    'Prijsbasis': 'per m2',
                    'MatchHints': ['baksteen', 'wildverband'],
                },
            ],
        },
    ]


def test_find_extracted_post_by_description_falls_back_to_fuzzy_match():
    offer_result = {
        'Bestand': 'offer.pdf',
        'Posten': [
            {'Omschrijving': 'Vermetselen gevelsteen halfsteens verband rood', 'Totaalbedrag': '500'},
        ],
    }

    # A slightly reworded/truncated copy instead of a literal one still resolves.
    post = find_extracted_post_by_description(offer_result, 'Vermetselen gevelsteen halfsteens verband')

    assert post.get('Totaalbedrag') == '500'


def test_find_extracted_posts_for_match_prefers_code_over_mismatched_description():
    offer_result = {
        'Bestand': 'offer.pdf',
        'Posten': [
            {'Omschrijving': 'Metselwerk gevel', 'Code': '44.31.10-a', 'Totaalbedrag': '1000'},
            {'Omschrijving': 'Voegwerk gevel', 'Code': '44.31.20-b', 'Totaalbedrag': '400'},
        ],
    }
    # The LLM's copied description doesn't match anything verbatim or fuzzily,
    # but the code it echoed back should still resolve the right post.
    offer_match = {
        'Match type': 'single',
        'Gematchte omschrijving': 'Iets heel anders',
        'Gematchte code': '44.31.10-A',
    }

    extracted_posts = find_extracted_posts_for_match(offer_result, offer_match)

    assert len(extracted_posts) == 1
    assert extracted_posts[0]['Totaalbedrag'] == '1000'


def test_complete_offer_info_flags_unlinked_claimed_match():
    offer_result = {
        'Bestand': 'offer.pdf',
        'Posten': [
            {'Omschrijving': 'Metselwerk gevel', 'Totaalbedrag': '1000'},
        ],
    }
    # LLM claims a match but the description is unresolvable and no code was given.
    offer_match = {
        'Match type': 'single',
        'Gematchte omschrijving': 'Volledig ongerelateerde post die niet bestaat',
        'Totaalbedrag': '9999',
    }

    info = complete_offer_info(offer_result, offer_match)

    assert info['Ongekoppeld'] is True
    assert 'niet worden teruggevonden' in ' '.join(warnings_for_offer({}, info))


def test_complete_offer_info_does_not_flag_no_match_claimed():
    offer_result = {'Bestand': 'offer.pdf', 'Posten': []}
    offer_match = {'Match type': 'single', 'Gematchte omschrijving': 'ONBEKEND'}

    info = complete_offer_info(offer_result, offer_match)

    assert info.get('Ongekoppeld') is not True
    assert warnings_for_offer({}, info) == []


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


def test_recalculate_matched_posts_derives_unit_price_from_extract_total_and_amount():
    comparison = {
        'MatchedPosten': [
            {
                'Omschrijving': 'Vloerwerk',
                'Aantal': '4',
                'Eenheid': 'm2',
                'Offertes': {
                    'solitas.pdf': {
                        'Gematchte omschrijving': 'PVC stroken',
                    },
                },
            },
        ],
    }
    offer_results = [
        {
            'Bestand': 'solitas.pdf',
            'Posten': [
                {
                    'Omschrijving': 'PVC stroken',
                    'Categorie': 'PVC',
                    'Aantal': '10',
                    'Eenheid': 'm2',
                    'Eenheidsprijs': 'ONBEKEND',
                    'Totaalbedrag': '250.00',
                },
            ],
        },
    ]
    matcher = ComparisonMatcher(folder_handler=None)
    matcher.project_offer_results = lambda _project: offer_results

    result = matcher.recalculate_matched_posts(comparison, project=object())
    offer = result[0]['Offertes']['solitas.pdf']

    assert offer['Eenheidsprijs'] == '25.00'
    assert offer['Totaalbedrag'] == '100.00'


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


def test_group_match_sums_amount_when_units_are_equal():
    comparison = {
        'Posten': [
            {
                'Omschrijving': 'Vloerwerk',
                'Aantal': '30',
                'Eenheid': 'm2',
            },
        ],
    }
    response = {
        'MatchedPosten': [
            {
                'Omschrijving': 'Vloerwerk',
                'Offertes': {
                    'solitas.pdf': {
                        'Match type': 'group',
                        'Gematchte posten': ['PVC 1', 'PVC 2'],
                    },
                },
            },
        ],
    }
    offer_results = [
        {
            'Bestand': 'solitas.pdf',
            'Posten': [
                {'Omschrijving': 'PVC 1', 'Aantal': '10', 'Eenheid': 'm2', 'Totaalbedrag': '100'},
                {'Omschrijving': 'PVC 2', 'Aantal': '20', 'Eenheid': 'm2', 'Totaalbedrag': '200'},
            ],
        },
    ]

    completed = complete_response(response, offer_results)
    normalized = normalize_matched_posts(comparison, completed, offer_results)
    offer = normalized[0]['Offertes']['solitas.pdf']

    assert offer['Aantal'] == '30.00'
    assert offer['Gematchte hoeveelheid eenheid'] == 'm2'


def test_match_response_accepts_offer_list_shape():
    offer_results = [
        {
            'Bestand': 'postma.pdf',
            'Posten': [
                {
                    'Omschrijving': 'Gevelsteen basis halfsteens verband',
                    'Categorie': 'Metselwerk',
                    'Eenheid': 'dzd',
                    'Eenheidsprijs': '765.00',
                    'Totaalbedrag': 'ONBEKEND',
                },
            ],
        },
    ]
    response = {
        'MatchedPosten': [
            {
                'Omschrijving': 'Vermetselen gevelsteen wildverband',
                'Offertes': [
                    {
                        'Bestand': 'postma.pdf',
                        'Match type': 'single',
                        'Gematchte omschrijving': 'Gevelsteen basis halfsteens verband',
                        'Gematchte posten': ['Gevelsteen basis halfsteens verband'],
                        'Overeenkomst': '2',
                    },
                ],
            },
        ],
    }
    comparison = {
        'Posten': [
            {
                'Omschrijving': 'Vermetselen gevelsteen wildverband',
                'Aantal': '86,95',
                'Eenheid': 'dzd',
            },
        ],
    }

    completed = complete_response(response, offer_results)
    normalized = normalize_matched_posts(comparison, completed, offer_results)

    offer = normalized[0]['Offertes']['postma.pdf']
    assert offer['Gematchte omschrijving'] == 'Gevelsteen basis halfsteens verband'
    assert offer['Gematchte categorie'] == 'Metselwerk'
    assert offer['Gematchte eenheid'] == 'dzd'
    assert offer['Eenheidsprijs'] == '765.00'
    assert offer['Overeenkomst'] == '2'


def test_calculate_offer_total_uses_total_when_comparison_unit_is_post():
    assert calculate_offer_total(
        '1',
        {
            'Gematchte eenheid': 'm1',
            'Eenheidsprijs': '12.00',
            'Totaalbedrag': '4800.00',
        },
        comparison_unit='post',
    ) == '4800.00'


def test_units_mismatch_normalizes_common_unit_spellings():
    assert units_mismatch('m2', 'm²') is False
    assert units_mismatch('m1', 'm¹') is False
    assert units_mismatch('dzd', 'st') is True
    assert units_mismatch('dzd', 'ONBEKEND') is False


def test_warning_for_offer():
    warning = warning_for_offer(
        {'Eenheid': 'dzd'},
        {'Gematchte eenheid': 'st', 'Overeenkomst': '1'},
    )

    assert 'Eenheid wijkt af' in warning
    assert 'Lage overeenkomstscore' in warning

    warning = warning_for_offer(
        {'Eenheid': 'dzd'},
        {'Gematchte eenheid': 'dzd', 'Overeenkomst': '2'},
    )

    assert 'Lage overeenkomstscore (2)' in warning


def test_warning_for_offer_allows_post_mismatch_with_total():
    warning = warning_for_offer(
        {'Eenheid': 'post'},
        {
            'Gematchte eenheid': 'm1',
            'Totaalbedrag': '4800.00',
            'Overeenkomst': '3',
        },
    )

    assert 'Eenheid wijkt af' not in warning


def test_warning_for_offer_keeps_post_mismatch_without_total():
    warning = warning_for_offer(
        {'Eenheid': 'post'},
        {
            'Gematchte eenheid': 'm1',
            'Totaalbedrag': 'ONBEKEND',
            'Overeenkomst': '3',
        },
    )

    assert 'Eenheid wijkt af' in warning


def test_warning_for_offer_warns_for_quantity_difference():
    warning = warning_for_offer(
        {'Aantal': '100', 'Eenheid': 'm2'},
        {'Aantal': '120', 'Gematchte eenheid': 'm2', 'Overeenkomst': '3'},
    )

    assert 'Hoeveelheid wijkt af' in warning


def test_warning_for_offer_allows_small_quantity_difference():
    warning = warning_for_offer(
        {'Aantal': '100', 'Eenheid': 'm2'},
        {'Aantal': '104', 'Gematchte eenheid': 'm2', 'Overeenkomst': '3'},
    )

    assert 'Hoeveelheid wijkt af' not in warning


def test_matched_post_options_exclude_posts_selected_in_other_rows():
    match_rows = [
        {
            'Omschrijving': 'Rij 1',
            'Offertes': {
                'offer.pdf': {'Gematchte omschrijving': 'Post A'},
            },
        },
        {
            'Omschrijving': 'Rij 2',
            'Offertes': {
                'offer.pdf': {'Gematchte omschrijving': 'Post B'},
            },
        },
    ]
    table = MatchedPostenTable(
        offer_names=['offer.pdf'],
        match_rows=match_rows,
        offer_post_descriptions={'offer.pdf': ['Post A', 'Post B', 'Post C']},
    )

    options = table.rows[0]['offer_0_omschrijving_options']

    assert 'Post A' in options
    assert 'Post B' not in options
    assert 'Post C' in options


def test_update_matched_cell_selects_extracted_offer_post():
    class FakeMatcher:
        def project_offer_results(self, _project):
            return [
                {
                    'Bestand': 'offer.pdf',
                    'Posten': [
                        {
                            'Omschrijving': 'Post A',
                            'Categorie': 'Cat',
                            'Aantal': '5',
                            'Eenheid': 'm2',
                            'Eenheidsprijs': 'ONBEKEND',
                            'Totaalbedrag': '100',
                        },
                    ],
                },
            ]

    class FakeProject:
        def save_comparison(self, comparison):
            self.saved_comparison = comparison

    service = ComparisonService(folder_handler=None, matcher=FakeMatcher())
    project = FakeProject()
    comparison = {}
    match_rows = [
        {
            'Omschrijving': 'Vergelijking',
            'Aantal': '2',
            'Eenheid': 'm2',
            'Offertes': {'offer.pdf': {}},
        },
    ]

    assert service.update_matched_cell(
        project,
        comparison,
        match_rows,
        0,
        'offer_0_omschrijving',
        'Post A',
        ['offer.pdf'],
    )

    offer = match_rows[0]['Offertes']['offer.pdf']
    assert offer['Gematchte omschrijving'] == 'Post A'
    assert offer['Gematchte categorie'] == 'Cat'
    assert offer['Gematchte eenheid'] == 'm2'
    assert offer['Eenheidsprijs'] == '20.00'
    assert offer['Totaalbedrag'] == '40.00'


def test_update_matched_cell_selects_multiple_extracted_offer_posts_as_group():
    class FakeMatcher:
        def project_offer_results(self, _project):
            return [
                {
                    'Bestand': 'offer.pdf',
                    'Posten': [
                        {
                            'Omschrijving': 'Post A',
                            'Categorie': 'Cat',
                            'Aantal': '5',
                            'Eenheid': 'm2',
                            'Eenheidsprijs': '20',
                            'Totaalbedrag': '100',
                        },
                        {
                            'Omschrijving': 'Post B',
                            'Categorie': 'Cat',
                            'Aantal': '10',
                            'Eenheid': 'm2',
                            'Eenheidsprijs': '20',
                            'Totaalbedrag': '200',
                        },
                    ],
                },
            ]

    class FakeProject:
        def save_comparison(self, comparison):
            self.saved_comparison = comparison

    service = ComparisonService(folder_handler=None, matcher=FakeMatcher())
    project = FakeProject()
    comparison = {}
    match_rows = [
        {
            'Omschrijving': 'Vergelijking',
            'Aantal': '1',
            'Eenheid': 'post',
            'Offertes': {'offer.pdf': {}},
        },
    ]

    assert service.update_matched_cell(
        project,
        comparison,
        match_rows,
        0,
        'offer_0_omschrijving',
        ['Post A', 'Post B'],
        ['offer.pdf'],
    )

    offer = match_rows[0]['Offertes']['offer.pdf']
    assert offer['Match type'] == 'group'
    assert offer['Gematchte omschrijving'] == '2 posten'
    assert offer['Gematchte posten'] == ['Post A', 'Post B']
    assert offer['Gematchte categorie'] == 'Cat'
    assert offer['Gematchte categorieen'] == ['Cat']
    assert offer['Gematchte eenheid'] == 'post'
    assert offer['Totaalbedrag'] == '300.00'


def test_selected_descriptions_accepts_json_array_string():
    assert ComparisonService.selected_descriptions('["Post A", "Post B"]') == ['Post A', 'Post B']


def test_seed_comparison_from_offer_prefills_rows_and_skips_structural_posts():
    class FakeMatcher:
        def project_offer_results(self, _project):
            return [
                {
                    'Bestand': 'offer.pdf',
                    'Posten': [
                        {'Omschrijving': 'Metselwerk gevel', 'Aantal': '10', 'Eenheid': 'm2'},
                        {'Omschrijving': 'Subtotaal', 'PostType': 'subtotal', 'Aantal': 'ONBEKEND', 'Eenheid': 'ONBEKEND'},
                        {'Omschrijving': 'ONBEKEND'},
                        {'Omschrijving': 'Voegwerk gevel', 'Aantal': 'ONBEKEND', 'Eenheid': 'ONBEKEND'},
                    ],
                },
            ]

    class FakeProject:
        def save_comparison(self, comparison):
            self.saved_comparison = comparison

    service = ComparisonService(folder_handler=None, matcher=FakeMatcher())
    project = FakeProject()
    comparison = {'Posten': [{'Omschrijving': 'Bestaande regel', 'Aantal': '', 'Eenheid': ''}]}

    added = service.seed_comparison_from_offer(project, comparison, 'offer.pdf')

    assert added == 2
    assert comparison['Posten'] == [
        {'Omschrijving': 'Bestaande regel', 'Aantal': '', 'Eenheid': ''},
        {'Omschrijving': 'Metselwerk gevel', 'Aantal': '10', 'Eenheid': 'm2'},
        {'Omschrijving': 'Voegwerk gevel', 'Aantal': '', 'Eenheid': ''},
    ]


def test_seed_comparison_from_offer_returns_zero_for_unknown_offer():
    class FakeMatcher:
        def project_offer_results(self, _project):
            return []

    service = ComparisonService(folder_handler=None, matcher=FakeMatcher())
    comparison = {}

    assert service.seed_comparison_from_offer(object(), comparison, 'missing.pdf') == 0
    assert comparison == {}
