from langchain_core.messages import HumanMessage, SystemMessage

from lc_patterns.models.chat import get_nova_pro


def main():
    print("🎭 Understanding Message Types\n")

    nova_pro = get_nova_pro()

    # Using structured messages for better control
    messages = [
        SystemMessage(content="You are a Databricks MLOps architect. Be concise and technical."),
        HumanMessage(content="What role does MLflow play in the Databricks ML lifecycle?"),
    ]

    response = nova_pro.invoke(messages)

    print("🤖 AI Response:\n")
    print(response.content)
    print("\n✅ Notice how the SystemMessage influenced the response style!")

if __name__ == "__main__":
    main()