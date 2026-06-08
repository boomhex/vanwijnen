from dataclasses import dataclass, field

@dataclass
class ComparisonPost:
    omschrijving: str = ''
    aantal: str = ''
    eenheid: str = ''

@dataclass
class MatchedOffer:
    match_type: str = ''
    gematchte_omschrijving: str = ''
    gematchte_posten: list[str] = field(default_factory=list)
    gematchte_eenheid: str = ''
    eenheidsprijs: str = ''
    totaalbedrag: str = ''
    overeenkomst: str = ''

@dataclass
class MatchedPost:
    omschrijving: str = ''
    aantal: str = ''
    eenheid: str = ''
    offertes: dict[str, MatchedOffer] = field(default_factory=dict)

@dataclass
class Comparison:
    posten: list[ComparisonPost] = field(default_factory=list)
    matched_posten: list[MatchedPost] = field(default_factory=list)
