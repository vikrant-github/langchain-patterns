from lc_patterns.tools.customer import get_customer


def test_get_customer_returns_customer_details() -> None:
    result = get_customer.invoke({"customer_id": "C001"})

    assert result == "Customer: Acme Corp; Segment: Enterprise; Status: Active"


def test_get_customer_returns_not_found_for_unknown_customer() -> None:
    result = get_customer.invoke({"customer_id": "C999"})

    assert result == "Customer 'C999' was not found."
