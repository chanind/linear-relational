from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class Prompt:
    text: str
    answer: str
    subject: str
    subject_name: Optional[str] = None
    object_name: Optional[str] = None
    relation_name: Optional[str] = None
