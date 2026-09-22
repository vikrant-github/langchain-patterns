from langchain.agents import create_agent

from lc_patterns.middleware.logging import AgentLoggingMiddleware
from lc_patterns.models.chat import get_nova_2_lite
from lc_patterns.tools.customer import get_customer
from lc_patterns.tools.weather import get_weather


def main() -> None:
    agent = create_agent(
        model=get_nova_2_lite(),
        tools=[get_customer, get_weather],
        middleware=[AgentLoggingMiddleware()],
        system_prompt=(
            "You are a helpful customer-service assistant. "
            "Use the available tools when needed."
        ),
    )

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Check customer C001 and tell me their account status. "
                        "Also check the weather in Delhi."
                    ),
                }
            ]
        }
    )

    print("\nFinal Answer:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()