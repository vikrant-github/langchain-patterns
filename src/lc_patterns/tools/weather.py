"""Weather tool for Chapter 4 function-calling examples."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    """Input schema for the weather tool."""

    city: str = Field(description="City name")


@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    temperatures = {
        "Seattle": 62,
        "Paris": 18,
        "Tokyo": 24,
    }

    temperature = temperatures.get(city, 72)
    return f"Current temperature in {city}: {temperature}°F"