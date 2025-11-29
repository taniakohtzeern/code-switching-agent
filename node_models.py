from typing import TypedDict, Optional, Annotated
from operator import add

class TranslationResponse(TypedDict):
    hypo: str

class AccuracyResponse(TypedDict):
    accuracy_score:float
    errors: dict[str, str]
    summary: str

class FluencyResponse(TypedDict):
    fluency_score: float
    errors: dict[str, str]
    summary: str


class NaturalnessResponse(TypedDict):
    naturalness_score: float
    observations: dict[str, str]
    summary: str


class CSRatioResponse(TypedDict):
    ratio_score: float
    computed_ratio: str
    notes: str


class SocialCulturalResponse(TypedDict):
    socio_cultural_score: float
    issues: str
    summary: str

class AgentRunningState(TypedDict):
    cs_ratio: str
    first_language: str
    second_language: str
    response: str
    data_generation_result: list[str]
    news_generation_result: list[str]

    fluency_result: FluencyResponse
    naturalness_result: NaturalnessResponse
    cs_ratio_result: CSRatioResponse
    social_cultural_result: SocialCulturalResponse

    summary: str
    score: float

    refine_count: Annotated[int, add]