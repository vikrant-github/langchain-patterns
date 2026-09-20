import time

from lc_patterns.models.chat import get_nova_2_lite, get_nova_pro


def compare_models():
    print("🔬 Comparing Bedrock Models\n")

    prompt = (
        "What are the main responsibilities of MLflow in a Databricks "
        "MLOps lifecycle? Limit your answer to 100 words."
    )

    models = {
        "Nova 2 Lite": get_nova_2_lite(),
        "Nova Pro": get_nova_pro(),
    }

    for model_name, model in models.items():
        print(f"\n📊 Testing: {model_name}")
        print("─" * 50)

        start_time = time.time()
        response = model.invoke(prompt)
        duration = (time.time() - start_time) * 1000

        print(f"Response: {response.content}")
        print(f"⏱️ Time: {duration:.0f}ms")


if __name__ == "__main__":
    compare_models()