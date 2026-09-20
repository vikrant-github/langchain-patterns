"""Demonstrate temperature parameter behavior with Databricks MLOps."""

from lc_patterns.models.chat import get_nova_2_lite


def print_token_usage(response):
    usage = response.usage_metadata
    if usage:
        print(
            f"  Tokens: input={usage.get('input_tokens', 'N/A')}, "
            f"output={usage.get('output_tokens', 'N/A')}, "
            f"total={usage.get('total_tokens', 'N/A')}"
        )
    else:
        print("  ⚠️ Token usage information unavailable.")


def temperature_comparison():
    prompt = "Explain Databricks Asset Bundles in one sentence."
    temperatures = [0.0, 0.5, 1.0]

    for temperature in temperatures:
        print(f"\n🌡️ Temperature: {temperature}")
        print("-" * 60)

        model = get_nova_2_lite().bind(
            temperature=temperature,
            max_tokens=500,
        )

        try:
            for attempt in range(1, 3):
                response = model.invoke(prompt)
                print(f"  Try {attempt}: {response.content}")
                print_token_usage(response)

        except Exception as error:  # noqa: BLE001
            print(f"  ⚠️ Temperature {temperature} is not supported.")
            print(f"  💡 Error: {error}")

    print("\n💡 General Temperature Guidelines:")
    print("   - Lower values: More deterministic and consistent responses")
    print("   - Medium values: Balanced creativity and consistency")
    print("   - Higher values: More creative and varied responses")


if __name__ == "__main__":
    temperature_comparison()