import pytest
from pydantic import ValidationError

from lc_patterns.prompts.templates import (
    chat_template,
    create_few_shot_template,
    simple_template,
)
from lc_patterns.schemas.structured_outputs import GovernedModel, ModelMetadata


def test_model_metadata_valid_construction() -> None:
    metadata = ModelMetadata(
        name="Churn Prediction",
        version="3.2",
        framework="XGBoost",
        purpose="Predict customer attrition",
    )

    assert metadata.name == "Churn Prediction"
    assert metadata.version == "3.2"
    assert metadata.framework == "XGBoost"
    assert metadata.purpose == "Predict customer attrition"


def test_governed_model_valid_construction() -> None:
    governed_model = GovernedModel(
        model=ModelMetadata(
            name="Customer Churn Prediction",
            version="3.2",
            framework="XGBoost",
            purpose="Predict customer attrition",
        ),
        governance={
            "lifecycle_stage": "production",
            "owner": {
                "team": "Customer Analytics",
                "contact": "Tony Stark",
            },
            "approval_required": True,
        },
    )

    assert governed_model.model.name == "Customer Churn Prediction"
    assert governed_model.governance.lifecycle_stage == "production"
    assert governed_model.governance.owner.team == "Customer Analytics"
    assert governed_model.governance.owner.contact == "Tony Stark"
    assert governed_model.governance.approval_required is True


def test_invalid_lifecycle_stage_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="lifecycle_stage"):
        GovernedModel(
            model=ModelMetadata(
                name="Customer Churn Prediction",
                version="3.2",
                framework="XGBoost",
                purpose="Predict customer attrition",
            ),
            governance={
                "lifecycle_stage": "retired",
                "owner": {
                    "team": "Customer Analytics",
                    "contact": "Tony Stark",
                },
                "approval_required": True,
            },
        )


def test_chat_template_input_variables_and_rendering() -> None:
    assert chat_template.input_variables == ["question", "role"]

    rendered = chat_template.format(question="What is ML?", role="ML engineer")

    assert "You are a helpful ML engineer." in rendered
    assert "What is ML?" in rendered


def test_simple_template_input_variables_and_rendering() -> None:
    assert simple_template.input_variables == ["question", "topic"]

    rendered = simple_template.format(
        question="What is ML?",
        topic="machine learning",
    )

    assert rendered == "Answer this machine learning question briefly: What is ML?"


def test_few_shot_template_construction() -> None:
    examples = [
        {
            "input": "What is a model?",
            "output": "A model is a trained artifact.",
        },
        {
            "input": "What is a view?",
            "output": "A view is a virtual table.",
        },
    ]

    few_shot_template = create_few_shot_template(examples)

    rendered = few_shot_template.format_messages(input="What is a table?")

    assert few_shot_template.examples == examples
    assert len(rendered) == 4
    assert rendered[0].content == "What is a model?"
    assert rendered[1].content == "A model is a trained artifact."
    assert rendered[2].content == "What is a view?"
    assert rendered[3].content == "A view is a virtual table."
