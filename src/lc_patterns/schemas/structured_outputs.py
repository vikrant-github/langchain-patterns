from typing import Literal

from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Schema for structured ML model metadata."""

    name: str = Field(description="Registered model name")
    version: str = Field(description="Model version")
    framework: str = Field(description="Machine learning framework")
    purpose: str = Field(description="Primary purpose of the model")


class ModelOwner(BaseModel):
    """Schema for ML model ownership information."""

    team: str = Field(description="Team responsible for the model")
    contact: str = Field(description="Primary contact for the model")


class ModelGovernance(BaseModel):
    """Schema for ML model governance information."""

    lifecycle_stage: Literal["development", "staging", "production"] = Field(
        description="Current lifecycle stage of the model"
    )
    owner: ModelOwner = Field(description="Model ownership information")
    approval_required: bool = Field(
        description="Whether approval is required before production deployment"
    )


class GovernedModel(BaseModel):
    """Schema for structured ML model and governance information."""

    model: ModelMetadata = Field(description="ML model metadata")
    governance: ModelGovernance = Field(
        description="ML model governance information"
    )