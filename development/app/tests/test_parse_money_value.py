from services.extract_offer import (
    build_posts_chunk_prompt,
    filter_non_price_posts,
    merge_post_chunks,
    parse_offer_response,
    parse_money_value,
    parse_posts_response,
    recover_complete_posts,
    recover_json_string_field,
    split_text_chunks,
)
from domain.offer import Posten
from domain.money import format_money
from decimal import Decimal


def test_parse_money_value():
    assert parse_money_value('1200,0') == Decimal("1200.00")
    assert parse_money_value('1200,-') == Decimal("1200.00")
    assert parse_money_value('1200.0') == Decimal("1200")
    assert parse_money_value('1200,-') == Decimal("1200.00")


def test_format_money_without_currency_symbol():
    assert format_money(Decimal('1234.5')) == '1.234,50'


def test_parse_posts_response_valid_json():
    posts, recovered = parse_posts_response('{"Posten": [{"Omschrijving": "A"}]}')

    assert posts == [{'Omschrijving': 'A'}]
    assert recovered is False


def test_posten_preserves_extra_matching_fields():
    post = Posten.from_dict({
        'Omschrijving': 'Vermetselen WF handvorm halfsteens verband',
        'Beschrijving': 'Basis halfsteens verband',
        'Categorie': 'Metselwerk',
        'Aantal': 'ONBEKEND',
        'Eenheid': 'dzd',
        'Eenheidsprijs': '800.00',
        'Totaalbedrag': 'ONBEKEND',
        'PostType': 'unit_rate',
        'Werksoort': 'metselwerk',
        'MatchHints': ['WF', 'halfsteens verband'],
    })

    assert post.to_dict()['PostType'] == 'unit_rate'
    assert post.to_dict()['Werksoort'] == 'metselwerk'
    assert post.to_dict()['MatchHints'] == ['WF', 'halfsteens verband']


def test_recover_complete_posts_from_truncated_json():
    response = '''{
      "Posten": [
        {"Omschrijving": "A", "Aantal": "1"},
        {"Omschrijving": "B", "Aantal": "2"},
        {"Omschrijving": "C", "Aantal": "'''

    assert recover_complete_posts(response) == [
        {'Omschrijving': 'A', 'Aantal': '1'},
        {'Omschrijving': 'B', 'Aantal': '2'},
    ]

    posts, recovered = parse_posts_response(response)
    assert posts == [
        {'Omschrijving': 'A', 'Aantal': '1'},
        {'Omschrijving': 'B', 'Aantal': '2'},
    ]
    assert recovered is True


def test_parse_offer_response_recovers_complete_posts_from_truncated_json():
    response = '''{
      "Naam aannemer": "Van Wijnen",
      "Totaalprijs inc. BTW": "121,00",
      "Totaalprijs exc. BTW": "100,00",
      "Posten": [
        {"Omschrijving": "A", "Aantal": "1"},
        {"Omschrijving": "B", "Aantal": "2"},
        {"Omschrijving": "C", "Aantal": "'''

    offer, recovered = parse_offer_response(response)

    assert recovered is True
    assert offer['Naam aannemer'] == 'Van Wijnen'
    assert offer['Totaalprijs inc. BTW'] == '121,00'
    assert offer['Totaalprijs exc. BTW'] == '100,00'
    assert offer['Posten'] == [
        {'Omschrijving': 'A', 'Aantal': '1'},
        {'Omschrijving': 'B', 'Aantal': '2'},
    ]


def test_recover_json_string_field_decodes_escaped_values():
    assert recover_json_string_field('{"Naam aannemer": "A \\"quoted\\" name"}', 'Naam aannemer') == 'A "quoted" name'


def test_build_posts_chunk_prompt_without_previous_post():
    prompt = build_posts_chunk_prompt('BASE PROMPT', 'chunk text', index=1, total_chunks=3)

    assert 'BASE PROMPT' in prompt
    assert 'VORIGE GEACCEPTEERDE POST:\nGEEN' in prompt
    assert 'De chunking is technisch en is GEEN onderdeel van de offerte.' in prompt
    assert 'Deel 1 van 3:' in prompt
    assert 'chunk text' in prompt


def test_build_posts_chunk_prompt_includes_previous_post_context():
    previous_post = {
        'Omschrijving': 'Vorige post',
        'Totaalbedrag': '100,00',
    }

    prompt = build_posts_chunk_prompt(
        'BASE PROMPT',
        'next chunk text',
        index=2,
        total_chunks=3,
        previous_post=previous_post,
    )

    assert '"Omschrijving": "Vorige post"' in prompt
    assert '"Totaalbedrag": "100,00"' in prompt
    assert 'Neem de vorige geaccepteerde post niet opnieuw op.' in prompt
    assert 'Gebruik labels zoals "Deel", "VORIGE GEACCEPTEERDE POST" of "INSTRUCTIE VOOR OVERLAP" nooit als omschrijving, categorie of postinformatie.' in prompt
    assert 'Deel 2 van 3:' in prompt


def test_filter_non_price_posts_removes_bepalingen_without_price():
    posts = [
        {
            'Omschrijving': 'Toeslag per 10mm extra opzetten van kantelaven',
            'Categorie': 'Algemene uitgangspunten stukadoorswerk',
            'Eenheidsprijs': '1,65',
            'Totaalbedrag': '1,65',
            'Aantal': 'ONBEKEND',
        },
        {
            'Omschrijving': 'Indien wij genoodzaakt zijn dikker stucwerk aan te brengen zullen wij hier een toeslag voor berekenen.',
            'Categorie': 'ONBEKEND',
            'Eenheidsprijs': 'ONBEKEND',
            'Totaalbedrag': 'ONBEKEND',
            'Aantal': 'ONBEKEND',
        },
        {
            'Omschrijving': 'Offerte geldig tot 30 dagen na offertedatum.',
            'Categorie': 'Algemene voorwaarden',
            'Eenheidsprijs': 'ONBEKEND',
            'Totaalbedrag': 'ONBEKEND',
            'Aantal': 'ONBEKEND',
        },
    ]

    assert filter_non_price_posts(posts) == [posts[0]]


def test_split_text_chunks_uses_line_overlap():
    text = '\n'.join(f'line {index}' for index in range(1, 11))

    chunks = split_text_chunks(text, max_chars=28, overlap_lines=2)

    assert len(chunks) > 1
    assert chunks[0].splitlines()[-2:] == chunks[1].splitlines()[:2]


def test_split_text_chunks_progresses_with_large_overlap():
    text = '\n'.join(f'line {index}' for index in range(1, 6))

    chunks = split_text_chunks(text, max_chars=14, overlap_lines=99)

    assert chunks
    assert chunks[-1].splitlines()[-1] == 'line 5'


def test_merge_post_chunks_removes_overlap_duplicates_with_more_complete_row():
    first_chunk = [
        {
            'Omschrijving': '500010 Leveren en aanbrengen PVC-riool diam. 160 mm',
            'Categorie': 'HWA-RIOLERING DAKWATER',
            'Totaalbedrag': '3.363,75',
            'Eenheid': 'm1',
            'Eenheidsprijs': 'ONBEKEND',
            'Aantal': '117,00',
        },
    ]
    second_chunk = [
        {
            'Omschrijving': '500010 Leveren en aanbrengen PVC-riool diam. 160 mm t/m 200 mm',
            'Categorie': 'HWA-RIOLERING DAKWATER',
            'Totaalbedrag': '3.363,75',
            'Eenheid': 'm1',
            'Eenheidsprijs': '28,75',
            'Aantal': '117,00',
        },
    ]

    assert merge_post_chunks([first_chunk, second_chunk]) == [second_chunk[0]]


def test_merge_post_chunks_keeps_similar_rows_with_different_totals():
    chunks = [
        [
            {
                'Omschrijving': '500010 Leveren en aanbrengen PVC-riool diam. 160 mm',
                'Categorie': 'HWA-RIOLERING DAKWATER',
                'Totaalbedrag': '3.363,75',
                'Eenheid': 'm1',
                'Eenheidsprijs': '28,75',
                'Aantal': '117,00',
            },
            {
                'Omschrijving': '500020 Leveren en aanbrengen PVC-riool diam. 250 mm',
                'Categorie': 'HWA-RIOLERING DAKWATER',
                'Totaalbedrag': '386,00',
                'Eenheid': 'm1',
                'Eenheidsprijs': '38,60',
                'Aantal': '10,00',
            },
        ],
    ]

    assert merge_post_chunks(chunks) == chunks[0]


def test_merge_post_chunks_uses_code_field_over_guessed_description_prefix():
    # Different wording, no leading digits in the description, but the same
    # explicit Code field the model filled in — should still be recognized
    # as the same post across a chunk boundary.
    chunks = [
        [
            {
                'Omschrijving': 'Vermetselen spouwblad',
                'Categorie': 'Metselwerk',
                'Code': '44.31.10-a',
                'Totaalbedrag': 'ONBEKEND',
                'Eenheid': 'm2',
                'Eenheidsprijs': 'ONBEKEND',
                'Aantal': 'ONBEKEND',
            },
        ],
        [
            {
                'Omschrijving': 'Muren metselen buitenspouwblad',
                'Categorie': 'Metselwerk',
                'Code': '44.31.10-a',
                'Totaalbedrag': '1.200,00',
                'Eenheid': 'm2',
                'Eenheidsprijs': '60,00',
                'Aantal': '20,00',
            },
        ],
    ]

    merged = merge_post_chunks(chunks)

    assert len(merged) == 1
    assert merged[0]['Totaalbedrag'] == '1.200,00'


def test_merge_post_chunks_keeps_rows_with_only_same_total():
    chunks = [
        [
            {
                'Omschrijving': '500010 Leveren en aanbrengen PVC-riool diam. 160 mm',
                'Categorie': 'HWA-RIOLERING DAKWATER',
                'Totaalbedrag': '900,00',
                'Eenheid': 'm1',
                'Eenheidsprijs': '30,00',
                'Aantal': '30,00',
            },
        ],
        [
            {
                'Omschrijving': '500020 Leveren en aanbrengen PVC-riool diam. 250 mm',
                'Categorie': 'HWA-RIOLERING DAKWATER',
                'Totaalbedrag': '900,00',
                'Eenheid': 'm1',
                'Eenheidsprijs': '45,00',
                'Aantal': '20,00',
            },
        ],
    ]

    assert merge_post_chunks(chunks) == [chunks[0][0], chunks[1][0]]
