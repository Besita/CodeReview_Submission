from openenv.core.env_server import Action, Observation, State
from typing import List, Optional, Literal
from pydantic import ConfigDict, Field , field_validator

class CodeReviewAction(Action):
    issues: List[str]  = Field(default_factory=list) #LLM returning 5–6 issues 
    severity: Literal["low", "medium", "high"] # Prevents invalid values like "critical", "HIGH", etc.
                                                #Makes reward logic easier later
    suggestion: str = ""
    reasoning: str = ""
    model_config = ConfigDict(extra="allow")   # ✅ ADD THIS LLM sometimes returns extra keys → no more validation errors
                                                # Makes your system robust to imperfect outputs
    @field_validator("issues", mode="before") # someimes llm outputs empty issues list → penalty Validation 
    def check_not_empty(cls, v):
        if not v:
            return ["unknown issue"]  # fallback instead of crash
        
        cleaned = [i.strip().lower() for i in v if isinstance(i, str)]
        cleaned = list(set(cleaned))  # remove duplicates
        return cleaned[:3]  # small safety cap
    
    @field_validator("severity", mode="before")
    def normalize_severity(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
            if v in ["low", "medium", "high"]:
                return v
        return "low"  # safe fallback

class CodeReviewObservation(Observation):
    code: str 
    score: float = 0.1
    feedback: str = ""    
    done: bool = False
    remaining_issues: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")

class CodeReviewState(State):
    episode_id: str
    done: bool = False

    task_id: Optional[str] = None
    task: Optional[dict] = None            # full task definition (expected issues, severity, etc.)
    code: str = ""
    remaining_issues: List[str] = Field(default_factory=list)
    previous_issues: List[str] = Field(default_factory=list)
    step_count: int = 0
    feedback: str = ""

'''class CodeReviewState(State):  # multi-step learning , reward shaping, memory of past actions
    task_id: Optional[str] = None
    code: str = ""
    remaining_issues: List[str] = Field(default_factory=list)
    previous_issues: List[str] = Field(default_factory=list)
    step_count: int = 0
    feedback: str = ""'''


