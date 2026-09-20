"""Demonstrate multi-turn conversations, streaming, and retry handling."""
from langchain_core.messages import HumanMessage, SystemMessage

from lc_patterns.models.chat import get_nova_2_lite


def main():
    print("💬 Multi-Turn Conversation Example\n")

    model_with_retry = get_nova_2_lite().with_retry(stop_after_attempt=3, wait_fixed=1)

    messages = [
        SystemMessage(content="You are a Databricks MLOps architect. Be concise and technical."),
        HumanMessage(content="What role does MLflow play in the Databricks ML lifecycle?"),
    ]

    print("👤 User: What role does MLflow play in the Databricks ML lifecycle?")

    try:
        response1 = model_with_retry.invoke(messages)
        print("\n🤖 AI:", response1.content)
        messages.append(response1)

        question2 = "How does MLflow Model Registry fit into that lifecycle?"
        print(f"\n👤 User: {question2}")
        messages.append(HumanMessage(content=question2))

        response2 = model_with_retry.invoke(messages)
        print("\n🤖 AI:", response2.content)
        messages.append(response2)

        question3 = "How would you structure this for a production MLOps pipeline? Answer in 5 bullets."
        print(f"\n👤 User: {question3}")
        messages.append(HumanMessage(content=question3))

        print("\n🤖 AI (streaming):")
        for chunk in model_with_retry.stream(messages):
            if isinstance(chunk.content, str):
                print(chunk.content, end="", flush=True)
            elif isinstance(chunk.content, list):
                for block in chunk.content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        if text:
                            print(text, end="", flush=True)

        print("\n\n✅ Conversation context maintained.")
        print(f"📊 Messages in history: {len(messages)}")
    except Exception as error:  # noqa: BLE001
        print(f"\n❌ Conversation failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error):
            print("Authentication failed.")


if __name__ == "__main__":
    main()