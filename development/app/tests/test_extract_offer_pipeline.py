import json
from pathlib import Path

import services.extract_offer as extract_offer_module
from services.extract_offer import (
    OFFER_RESPONSE_SCHEMA,
    OFFER_SUMMARY_RESPONSE_SCHEMA,
    fetch_posts_chunk,
    find_post_page,
    friendly_error_message,
    has_extractable_text,
    load_prompt,
    normalize_amounts,
    normalize_units,
    read_pdf,
    read_pdf_with_pages,
    should_use_chunked_extraction,
    validate_offer_json,
)
from services.folder_handler import FolderHandler


def test_offer_schemas_include_btw_verlegd():
    assert 'BTW verlegd' in OFFER_RESPONSE_SCHEMA['properties']
    assert 'BTW verlegd' in OFFER_RESPONSE_SCHEMA['required']
    assert 'BTW verlegd' in OFFER_SUMMARY_RESPONSE_SCHEMA['properties']
    assert 'BTW verlegd' in OFFER_SUMMARY_RESPONSE_SCHEMA['required']


def test_validate_offer_json_does_not_flag_missing_btw_verlegd_on_old_data():
    # extract.json files saved before this field existed shouldn't suddenly
    # show a spurious "missing field" warning.
    offer = {
        'Naam aannemer': 'Test BV',
        'Totaalprijs inc. BTW': 'ONBEKEND',
        'Totaalprijs exc. BTW': 'ONBEKEND',
        'Posten': [],
    }

    warnings = validate_offer_json(offer)

    assert not any('BTW verlegd' in warning for warning in warnings)


def test_load_prompt_substitutes_shared_fragment(tmp_path, monkeypatch):
    shared_dir = tmp_path / '_shared'
    shared_dir.mkdir()
    (shared_dir / 'greeting.txt').write_text('Hello\nWorld\n')
    main_file = tmp_path / 'main.txt'
    main_file.write_text('Before\n{{SHARED:greeting}}\nAfter\n')

    monkeypatch.setattr(extract_offer_module, 'SHARED_PROMPTS_DIR', shared_dir)

    assert load_prompt(main_file) == 'Before\nHello\nWorld\nAfter\n'


def test_load_prompt_resolves_the_real_extraction_prompts(monkeypatch):
    app_dir = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(extract_offer_module, 'SHARED_PROMPTS_DIR', app_dir / 'prompts' / '_shared')

    one_shot = load_prompt(app_dir / 'prompts' / 'extract_prompt.txt')
    chunked = load_prompt(app_dir / 'prompts' / 'extract_posts_chunk_prompt.txt')

    assert '{{SHARED:' not in one_shot
    assert '{{SHARED:' not in chunked
    assert '"Omschrijving": "..."' in one_shot
    assert '"Omschrijving": "..."' in chunked
    # The shared rule text must appear in both, so a future edit to the
    # shared fragment can't silently only affect one extraction mode.
    assert '- Zet in Brontekst de relevante ruwe offertepassage' in one_shot
    assert '- Zet in Brontekst de relevante ruwe offertepassage' in chunked

    # stelpost / alternative-post-splitting rules (post_field_rules) must
    # reach both extraction modes.
    assert 'stelpost' in one_shot.casefold()
    assert 'stelpost' in chunked.casefold()
    assert 'PostType "alternative"' in one_shot
    assert 'PostType "alternative"' in chunked


def test_load_prompt_resolves_summary_rules_into_one_shot_and_summary_prompts(monkeypatch):
    app_dir = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(extract_offer_module, 'SHARED_PROMPTS_DIR', app_dir / 'prompts' / '_shared')

    one_shot = load_prompt(app_dir / 'prompts' / 'extract_prompt.txt')
    summary = load_prompt(app_dir / 'prompts' / 'extract_summary_prompt.txt')

    assert '{{SHARED:' not in summary
    for text in (one_shot, summary):
        assert '"BTW verlegd": "..."' in text
        assert 'Voor BTW verlegd gebruik je "ja"' in text
        assert 'gebruik dan die naam' in text


def test_fetch_posts_chunk_retries_and_keeps_the_more_complete_response(monkeypatch):
    truncated = '{"Posten": [{"Omschrijving": "A"}, {"Omschrijving": "B"'
    complete = '{"Posten": [{"Omschrijving": "A"}, {"Omschrijving": "B"}, {"Omschrijving": "C"}]}'
    answers = iter([truncated, complete])
    calls = []

    def fake_ask_llm(prompt, **kwargs):
        calls.append(kwargs.get('label'))
        return next(answers)

    monkeypatch.setattr(extract_offer_module, 'ask_llm', fake_ask_llm)

    posts, answer, recovered = fetch_posts_chunk(
        'RULES', 'chunk text', index=1, total_chunks=1, previous_post=None,
    )

    assert len(calls) == 2
    assert calls[1].endswith('_retry')
    assert recovered is False
    assert answer == complete
    assert [post['Omschrijving'] for post in posts] == ['A', 'B', 'C']


def test_fetch_posts_chunk_keeps_first_attempt_when_retry_recovers_fewer_posts(monkeypatch):
    truncated = '{"Posten": [{"Omschrijving": "A"}, {"Omschrijving": "B"}, {"Omschrijving": "C"'
    retry_worse = '{"Posten": [{"Omschrijving": "A"'  # only one *complete* post object
    answers = iter([truncated, retry_worse])

    monkeypatch.setattr(
        extract_offer_module, 'ask_llm', lambda prompt, **kwargs: next(answers),
    )

    posts, _answer, recovered = fetch_posts_chunk(
        'RULES', 'chunk text', index=1, total_chunks=1, previous_post=None,
    )

    assert recovered is True
    assert [post['Omschrijving'] for post in posts] == ['A', 'B']


def test_fetch_posts_chunk_keeps_first_attempt_when_retry_fails_to_parse_at_all(monkeypatch):
    truncated = '{"Posten": [{"Omschrijving": "A"}, {"Omschrijving": "B"'
    unparseable_retry = 'not even json'
    answers = iter([truncated, unparseable_retry])

    monkeypatch.setattr(
        extract_offer_module, 'ask_llm', lambda prompt, **kwargs: next(answers),
    )

    posts, answer, recovered = fetch_posts_chunk(
        'RULES', 'chunk text', index=1, total_chunks=1, previous_post=None,
    )

    assert recovered is True
    assert answer == truncated
    assert [post['Omschrijving'] for post in posts] == ['A']


def test_fetch_posts_chunk_saves_raw_response_before_a_hard_parse_failure(monkeypatch):
    # Not truncated-and-recoverable — genuinely invalid JSON, which
    # parse_posts_response cannot salvage at all and must raise for.
    unparseable = 'not even json'
    monkeypatch.setattr(extract_offer_module, 'ask_llm', lambda prompt, **kwargs: unparseable)

    saved = {}

    def response_callback(filename, content):
        saved[filename] = content

    try:
        fetch_posts_chunk(
            'RULES', 'chunk text', index=1, total_chunks=1, previous_post=None,
            response_callback=response_callback,
        )
        raised = False
    except ValueError:
        raised = True

    assert raised, 'expected parse_posts_response to raise on unparseable JSON'
    assert saved.get('llm_posts_chunk_1_response.txt') == unparseable


def test_fetch_posts_chunk_no_retry_when_response_is_complete(monkeypatch):
    complete = '{"Posten": [{"Omschrijving": "A"}]}'
    calls = []

    def fake_ask_llm(prompt, **kwargs):
        calls.append(kwargs.get('label'))
        return complete

    monkeypatch.setattr(extract_offer_module, 'ask_llm', fake_ask_llm)

    posts, _answer, recovered = fetch_posts_chunk(
        'RULES', 'chunk text', index=1, total_chunks=1, previous_post=None,
    )

    assert len(calls) == 1
    assert recovered is False
    assert [post['Omschrijving'] for post in posts] == ['A']


def test_friendly_error_message_for_password_protected_pdf():
    from pdfminer.pdfdocument import PDFPasswordIncorrect

    message = friendly_error_message(PDFPasswordIncorrect('incorrect password'))
    assert 'wachtwoord' in message.casefold()


def test_friendly_error_message_for_corrupt_pdf():
    from pdfminer.pdfparser import PDFSyntaxError

    message = friendly_error_message(PDFSyntaxError('bad xref table'))
    assert 'beschadigd' in message.casefold()


def test_friendly_error_message_for_quota_error():
    message = friendly_error_message(RuntimeError('429 RESOURCE_EXHAUSTED: quota exceeded'))
    assert 'limiet' in message.casefold()


def test_friendly_error_message_for_timeout():
    message = friendly_error_message(RuntimeError('Deadline Exceeded: the request timed out'))
    assert 'timeout' in message.casefold() or 'lang' in message.casefold()


def test_friendly_error_message_passes_through_own_value_errors():
    message = friendly_error_message(ValueError('Er is nauwelijks tekst uit deze PDF gehaald.'))
    assert message == 'Er is nauwelijks tekst uit deze PDF gehaald.'


def test_has_extractable_text_rejects_empty_or_near_empty():
    assert has_extractable_text('') is False
    assert has_extractable_text('   \n\n   ') is False
    assert has_extractable_text('1 2 3', min_chars=40) is False


def test_has_extractable_text_accepts_real_content():
    text = 'Offerte voor metselwerk\nVermetselen spouwblad\nTotaal 1.234,56 euro'
    assert has_extractable_text(text, min_chars=40) is True


def test_normalize_units_maps_known_synonyms():
    offer = {
        'Posten': [
            {'Omschrijving': 'A', 'Eenheid': 'stuks'},
            {'Omschrijving': 'B', 'Eenheid': 'm²'},
            {'Omschrijving': 'C', 'Eenheid': 'ONBEKEND'},
        ],
    }

    normalized = normalize_units(offer)

    assert normalized['Posten'][0]['Eenheid'] == 'st'
    assert normalized['Posten'][1]['Eenheid'] == 'm2'
    assert normalized['Posten'][2]['Eenheid'] == 'ONBEKEND'


def test_normalize_units_leaves_unrecognized_values_unchanged():
    offer = {'Posten': [{'Omschrijving': 'A', 'Eenheid': 'kg'}]}

    normalized = normalize_units(offer)

    assert normalized['Posten'][0]['Eenheid'] == 'kg'


def test_validate_offer_json_flags_unrecognized_unit():
    offer = {
        'Naam aannemer': 'ONBEKEND',
        'Totaalprijs inc. BTW': 'ONBEKEND',
        'Totaalprijs exc. BTW': 'ONBEKEND',
        'Posten': [{'Omschrijving': 'A', 'Eenheid': 'kg', 'Totaalbedrag': '10', 'Eenheidsprijs': 'ONBEKEND'}],
    }

    warnings = validate_offer_json(offer)

    assert any('Eenheid' in warning and 'kg' in warning for warning in warnings)


def test_validate_offer_json_accepts_canonical_unit():
    offer = {
        'Naam aannemer': 'ONBEKEND',
        'Totaalprijs inc. BTW': 'ONBEKEND',
        'Totaalprijs exc. BTW': 'ONBEKEND',
        'Posten': [{'Omschrijving': 'A', 'Eenheid': 'm2', 'Totaalbedrag': '10', 'Eenheidsprijs': 'ONBEKEND'}],
    }

    warnings = validate_offer_json(offer)

    assert not any('Eenheid' in warning for warning in warnings)


def test_normalize_amounts_converts_summary_and_post_fields():
    offer = {
        'Totaalprijs inc. BTW': '1.210,00',
        'Totaalprijs exc. BTW': '1.000,00',
        'Posten': [
            {'Omschrijving': 'A', 'Eenheidsprijs': '10,50', 'Totaalbedrag': '21,00', 'Aantal': '2'},
        ],
    }

    normalized = normalize_amounts(offer)

    assert normalized['Totaalprijs inc. BTW'] == '1210.00'
    assert normalized['Totaalprijs exc. BTW'] == '1000.00'
    assert normalized['Posten'][0]['Eenheidsprijs'] == '10.50'
    assert normalized['Posten'][0]['Totaalbedrag'] == '21.00'
    assert normalized['Posten'][0]['Aantal'] == '2'


def test_normalize_amounts_keeps_unknown_and_malformed_values_untouched():
    offer = {'Totaalprijs inc. BTW': 'ONBEKEND', 'Posten': [{'Omschrijving': 'A', 'Totaalbedrag': 'niet-een-bedrag'}]}

    normalized = normalize_amounts(offer)

    assert normalized['Totaalprijs inc. BTW'] == 'ONBEKEND'
    assert normalized['Posten'][0]['Totaalbedrag'] == 'niet-een-bedrag'


def test_should_use_chunked_extraction_respects_forced_modes(monkeypatch):
    monkeypatch.setattr(extract_offer_module, 'EXTRACTION_MODE', 'chunked')
    assert should_use_chunked_extraction('short') is True

    monkeypatch.setattr(extract_offer_module, 'EXTRACTION_MODE', 'one_shot')
    assert should_use_chunked_extraction('x' * 100_000) is False


def test_should_use_chunked_extraction_auto_uses_char_threshold(monkeypatch):
    monkeypatch.setattr(extract_offer_module, 'EXTRACTION_MODE', 'auto')
    monkeypatch.setattr(extract_offer_module, 'CHUNKED_EXTRACTION_THRESHOLD', 100)

    assert should_use_chunked_extraction('x' * 50) is False
    assert should_use_chunked_extraction('x' * 200) is True


def test_should_use_chunked_extraction_rejects_invalid_mode(monkeypatch):
    monkeypatch.setattr(extract_offer_module, 'EXTRACTION_MODE', 'sometimes')

    try:
        should_use_chunked_extraction('text')
        assert False, 'expected ValueError'
    except ValueError as error:
        assert 'EXTRACT_MODE' in str(error)


def _make_offer_document(tmp_path) -> tuple[Path, FolderHandler]:
    folder_handler = FolderHandler(tmp_path)
    project = folder_handler.create_project('TestProject')
    offer_dir = project.path / 'offer1'
    offer_dir.mkdir()
    document = offer_dir / 'document.pdf'
    document.write_bytes(b'%PDF-1.4 fake content, unused because read_pdf is mocked')
    return document, folder_handler


def test_extract_offer_one_shot_end_to_end(tmp_path, monkeypatch):
    document, folder_handler = _make_offer_document(tmp_path)

    monkeypatch.setattr(extract_offer_module, 'EXTRACTION_MODE', 'one_shot')
    page_text = 'Genoeg tekst voor een geldige offerte extractie test. Werk A Alternatief bij Werk A goedkoper materiaal'
    monkeypatch.setattr(extract_offer_module, 'read_pdf_with_pages', lambda file: (page_text, [page_text]))
    monkeypatch.setattr(extract_offer_module, 'load_prompt', lambda path: 'PROMPT')

    canned_answer = json.dumps({
        'Naam aannemer': 'Test BV',
        'Totaalprijs inc. BTW': 'ONBEKEND',
        'Totaalprijs exc. BTW': '100,00',
        'BTW verlegd': 'ja',
        'Posten': [
            {
                'Omschrijving': 'Werk A',
                'Categorie': 'Cat',
                'Totaalbedrag': '100,00',
                'Eenheid': 'stuks',
                'Eenheidsprijs': '100,00',
                'Aantal': '1',
            },
            {
                'Omschrijving': 'Alternatief bij Werk A: goedkoper materiaal',
                'PostType': 'alternative',
                'Categorie': 'Cat',
                'Totaalbedrag': '-9,30',
                'Eenheid': 'stuks',
                'Eenheidsprijs': 'ONBEKEND',
                'Aantal': 'ONBEKEND',
            },
        ],
    })
    monkeypatch.setattr(extract_offer_module, 'ask_llm', lambda prompt, **kwargs: canned_answer)

    result = extract_offer_module.extract_offer(document, folder_handler)

    assert result['Naam aannemer'] == 'Test BV'
    assert result['BTW verlegd'] == 'ja'
    assert result['Posten'][0]['Eenheid'] == 'st'
    assert result['Posten'][0]['Totaalbedrag'] == '100.00'
    assert result['Posten'][1]['PostType'] == 'alternative'
    assert result['Posten'][1]['Totaalbedrag'] == '-9.30'
    assert result['Posten'][0]['Pagina'] == '1'
    assert result['Posten'][1]['Pagina'] == '1'

    status = folder_handler.load_extraction_status(document)
    assert status['status'] == 'done'
    assert folder_handler.load_result(document) == result


def test_extract_offer_chunked_end_to_end(tmp_path, monkeypatch):
    document, folder_handler = _make_offer_document(tmp_path)

    monkeypatch.setattr(extract_offer_module, 'EXTRACTION_MODE', 'chunked')
    page_one = 'Genoeg tekst voor een geldige offerte extractie test, verder niets relevants hier.'
    page_two = 'Post 1 tekst met bijpassende woorden voor de paginamatch.'
    monkeypatch.setattr(
        extract_offer_module,
        'read_pdf_with_pages',
        lambda file: (f'{page_one}\n{page_two}', [page_one, page_two]),
    )
    monkeypatch.setattr(extract_offer_module, 'load_prompt', lambda path: 'PROMPT')

    def fake_ask_llm(prompt, **kwargs):
        label = kwargs.get('label', '')
        if label == 'offer_summary':
            return json.dumps({
                'Naam aannemer': 'Chunked BV',
                'Totaalprijs inc. BTW': '60,50',
                'Totaalprijs exc. BTW': '50,00',
            })
        return json.dumps({
            'Posten': [
                {
                    'Omschrijving': 'Post 1',
                    'Categorie': 'Cat',
                    'Totaalbedrag': '50,00',
                    'Eenheid': 'm²',
                    'Eenheidsprijs': '50,00',
                    'Aantal': '1',
                },
            ],
        })

    monkeypatch.setattr(extract_offer_module, 'ask_llm', fake_ask_llm)

    result = extract_offer_module.extract_offer(document, folder_handler)

    assert result['Naam aannemer'] == 'Chunked BV'
    assert result['Posten'][0]['Eenheid'] == 'm2'
    assert result['Posten'][0]['Totaalbedrag'] == '50.00'
    assert result['Posten'][0]['Pagina'] == '2'

    status = folder_handler.load_extraction_status(document)
    assert status['status'] == 'done'


def test_extract_offer_fails_fast_on_empty_pdf_text(tmp_path, monkeypatch):
    document, folder_handler = _make_offer_document(tmp_path)

    monkeypatch.setattr(extract_offer_module, 'read_pdf_with_pages', lambda file: ('', []))
    monkeypatch.setattr(extract_offer_module, 'load_prompt', lambda path: 'PROMPT')

    def fail_if_called(prompt, **kwargs):
        raise AssertionError('ask_llm should not be called for an unreadable PDF')

    monkeypatch.setattr(extract_offer_module, 'ask_llm', fail_if_called)

    try:
        extract_offer_module.extract_offer(document, folder_handler)
        assert False, 'expected ValueError'
    except ValueError:
        pass

    status = folder_handler.load_extraction_status(document)
    assert status['status'] == 'failed'
    assert 'scan' in status['error'].casefold()


def test_ask_llm_fails_fast_on_missing_api_key(monkeypatch):
    call_count = 0

    def fake_get_client():
        nonlocal call_count
        call_count += 1
        raise RuntimeError('GEMINI_API_KEY is not set')

    def fail_if_sleeps(seconds):
        raise AssertionError('ask_llm should not retry/sleep on a missing API key')

    monkeypatch.setattr(extract_offer_module, 'get_client', fake_get_client)
    monkeypatch.setattr(extract_offer_module.time, 'sleep', fail_if_sleeps)

    try:
        extract_offer_module.ask_llm('prompt')
        assert False, 'expected RuntimeError'
    except RuntimeError as error:
        assert 'GEMINI_API_KEY is not set' in str(error)

    assert call_count == 1


def test_read_pdf_with_pages_matches_pdfplumber_page_count_and_joined_text():
    import pdfplumber

    document = Path(__file__).resolve().parents[1] / 'storage' / 'test' / '22.31_baksteen' / 'postma' / 'document.pdf'

    joined, pages = read_pdf_with_pages(document)

    with pdfplumber.open(document) as pdf:
        assert len(pages) == len(pdf.pages)

    assert read_pdf(document) == joined
    assert any(pages), 'expected at least one page to contain extractable text'


def test_find_post_page_matches_exact_text():
    pages = [
        'Introductiepagina zonder relevante inhoud.',
        'Vermetselen gevelsteen halfsteens verband 765,00 per dzd',
    ]

    assert find_post_page('Vermetselen gevelsteen halfsteens verband', pages) == '2'


def test_find_post_page_matches_reworded_text():
    pages = [
        'Toeslag rond werk uitgevoerd in strekken 15,00 per M2',
        'Iets volledig anders op deze pagina.',
    ]

    # Not a literal copy, but shares most of the same words.
    assert find_post_page('Toeslag voor rond werk, uitgevoerd in strekken', pages) == '1'


def test_find_post_page_returns_onbekend_below_threshold():
    pages = ['Deze pagina heeft nauwelijks overlappende woorden.']

    assert find_post_page('Compleet andere post omschrijving hier', pages) == 'ONBEKEND'


def test_find_post_page_ties_go_to_the_earliest_page():
    pages = [
        'Regiewerk metselaar per uur',
        'Regiewerk metselaar per uur',
    ]

    assert find_post_page('Regiewerk metselaar per uur', pages) == '1'


def test_find_post_page_handles_empty_input():
    assert find_post_page('', ['Some page text']) == 'ONBEKEND'
    assert find_post_page('Some text', []) == 'ONBEKEND'
    assert find_post_page('ONBEKEND', ['Some page text']) == 'ONBEKEND'
