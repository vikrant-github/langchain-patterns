from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)

# Basic reusable chat template
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful {role}."),
        ("human", "{question}"),
    ]
)

# Basic reusable text template
simple_template = PromptTemplate.from_template(
    "Answer this {topic} question briefly: {question}"
)

# Reusable few-shot example template
few_shot_example_template = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)


def create_few_shot_template(
    examples: list[dict[str, str]],
) -> FewShotChatMessagePromptTemplate:
    """Create a reusable few-shot prompt from input/output examples."""
    return FewShotChatMessagePromptTemplate(
        example_prompt=few_shot_example_template,
        examples=examples,
    )