from langchain_core.tools import tool
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customer_id: str = Field(description="Unique customer identifier")


_CUSTOMERS = {
    "C001": {
        "name": "Acme Corp",
        "segment": "Enterprise",
        "status": "Active",
    },
    "C002": {
        "name": "Northstar Health",
        "segment": "Mid-Market",
        "status": "Active",
    },
}


@tool(args_schema=CustomerInput)
def get_customer(customer_id: str) -> str:
    """Retrieve customer information using a customer ID."""
    customer = _CUSTOMERS.get(customer_id)

    if customer is None:
        return f"Customer '{customer_id}' was not found."

    return (
        f"Customer: {customer['name']}; "
        f"Segment: {customer['segment']}; "
        f"Status: {customer['status']}"
    )