from domain.offer import Posten, grouping_fields


def make_post(categorie: str = '', **extra: str) -> Posten:
    return Posten(omschrijving='Post', categorie=categorie, extra=extra)


def test_grouping_fields_empty_for_flat_offer_with_unique_codes():
    # Postma/Coenen-style: one Categorie, a unique bestekcode per post —
    # Code must not be mistaken for a repeating group label.
    posten = [
        make_post('Metselwerk', Code='44.31.10-a'),
        make_post('Metselwerk', Code='44.31.20-b'),
        make_post('Metselwerk', Code='44.31.30-c'),
    ]

    assert grouping_fields(posten) == []


def test_grouping_fields_groups_by_categorie_only_when_codes_are_unique():
    posten = [
        make_post('A', Code='001'),
        make_post('A', Code='002'),
        make_post('B', Code='003'),
        make_post('B', Code='004'),
    ]

    assert grouping_fields(posten) == ['Categorie']


def test_grouping_fields_groups_by_categorie_and_repeated_regelnummer():
    # pwz-style: "V01"/"V02" repeat across several distinct posts within
    # each Categorie section, not a unique per-post identifier.
    posten = [
        make_post('Vloeren begane grond', Regelnummer='V01'),
        make_post('Vloeren begane grond', Regelnummer='V01'),
        make_post('Vloeren begane grond', Regelnummer='V02'),
        make_post('Vloeren begane grond', Regelnummer='V02'),
        make_post('Vloeren 1e etage', Regelnummer='V01'),
        make_post('Vloeren 1e etage', Regelnummer='V01'),
        make_post('Vloeren 1e etage', Regelnummer='V02'),
        make_post('Vloeren 1e etage', Regelnummer='V02'),
    ]

    assert grouping_fields(posten) == ['Categorie', 'Regelnummer']


def test_grouping_fields_ignores_a_single_incidental_duplicate():
    posten = [
        make_post('A', Code='001'),
        make_post('A', Code='001'),  # only one repeated value among these
        make_post('A', Code='002'),
        make_post('A', Code='003'),
        make_post('A', Code='004'),
    ]

    # Only one distinct value repeats and Categorie has no second value —
    # neither field should trigger grouping.
    assert grouping_fields(posten) == []


def test_grouping_fields_empty_for_no_posts():
    assert grouping_fields([]) == []
