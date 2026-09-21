from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from lc_patterns.models.chat import get_nova_2_lite
from lc_patterns.tools.customer import get_customer
from lc_patterns.tools.weather import get_weather


def main() -> None:
    agent = create_agent(
        model=get_nova_2_lite(),
        tools=[get_weather, get_customer],
        system_prompt=(
            "You are a helpful assistant. "
            "Use the available tools when they are needed to answer the user."
        ),
    )

    queries = [
        "What is the weather in Tokyo?",
        "Tell me about customer C001.",
    ]

    for query in queries:
        response = agent.invoke(
            {"messages": [HumanMessage(content=query)]}
        )

        print(f"User: {query}")
        print(f"Agent: {response['messages'][-1].content}")
        print()


if __name__ == "__main__":
    main()