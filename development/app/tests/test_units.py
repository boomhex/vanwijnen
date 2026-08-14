from domain.units import canonicalize_unit


def test_canonicalize_unit_recognizes_known_variants():
    assert canonicalize_unit('m2') == ('m2', True)
    assert canonicalize_unit('m²') == ('m2', True)
    assert canonicalize_unit('vierkante meter') == ('m2', True)
    assert canonicalize_unit('meter') == ('m1', True)
    assert canonicalize_unit('m¹') == ('m1', True)
    assert canonicalize_unit('stuks') == ('st', True)
    assert canonicalize_unit('Stk') == ('st', True)
    assert canonicalize_unit('dznd') == ('dzd', True)
    assert canonicalize_unit('stelpost') == ('post', True)


def test_canonicalize_unit_is_case_insensitive_and_strips_spaces():
    assert canonicalize_unit(' STUKS ') == ('st', True)


def test_canonicalize_unit_empty_and_unknown_are_not_flagged():
    assert canonicalize_unit(None) == ('', True)
    assert canonicalize_unit('') == ('', True)
    assert canonicalize_unit('ONBEKEND') == ('ONBEKEND', True)


def test_canonicalize_unit_flags_unrecognized_value():
    value, recognized = canonicalize_unit('kg')
    assert value == 'kg'
    assert recognized is False
