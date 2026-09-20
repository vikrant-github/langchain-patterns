"""Basic LangChain invocation.

get_nova_2_lite() returns a ChatBedrockConverse client configured for
Amazon Nova 2 Lite in us-east-1. invoke() sends the prompt and returns
an AI message; response.content contains the generated text.

Model configuration lives in src/lc_patterns/models/chat.py. Credentials
are resolved through the standard AWS SDK configuration chain.
"""

from lc_patterns.models.chat import get_nova_2_lite


def main():
    print("🦜🔗 Hello LangChain!\n")

    nova_2_lite = get_nova_2_lite()
    
    response = nova_2_lite.invoke("Explain the four core stages of an MLOps lifecycle and give one key engineering concern for each. Limit your answer to 120 words.")

    print("🤖 AI Response:", response.content)
    print("\n✅ Success! You just made your first LangChain call!")

if __name__ == "__main__":
    main()