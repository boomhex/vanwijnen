from services.extract_offer import parse_money_value
from decimal import Decimal


def test_parse_money_value():
    assert parse_money_value('1200,0') == Decimal("1200.00")
    assert parse_money_value('1200,-') == Decimal("1200.00")
    assert parse_money_value('1200.0') == Decimal("1200")
    assert parse_money_value('1200,-') == Decimal("1200.00")

