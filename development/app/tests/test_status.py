from domain.status import extraction_progress_fraction


def test_extraction_progress_fraction_unknown_step():
    assert extraction_progress_fraction(None) is None
    assert extraction_progress_fraction('') is None
    assert extraction_progress_fraction('some_unknown_step') is None


def test_extraction_progress_fraction_increases_through_known_steps():
    steps = [
        'reading_pdf',
        'saving_raw_text',
        'calling_llm',
        'saving_llm_response',
        'validating_json',
        'normalizing_amounts',
        'saving_extract',
    ]
    fractions = [extraction_progress_fraction(step) for step in steps]
    assert all(fraction is not None for fraction in fractions)
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_extraction_progress_fraction_treats_llm_call_aliases_the_same():
    assert extraction_progress_fraction('calling_llm') == extraction_progress_fraction('extracting_chunked')
    assert extraction_progress_fraction('extracting_summary') == extraction_progress_fraction('extracting_chunked')


def test_extraction_progress_fraction_for_chunked_posts_progresses_within_phase():
    first = extraction_progress_fraction('extracting_posts_chunk_1_of_4')
    third = extraction_progress_fraction('extracting_posts_chunk_3_of_4')
    last = extraction_progress_fraction('extracting_posts_chunk_4_of_4')

    before = extraction_progress_fraction('saving_raw_text')
    after = extraction_progress_fraction('saving_llm_response')

    assert before < first < third < last < after
