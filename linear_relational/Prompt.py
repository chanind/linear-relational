from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prompt:
    text: str
    answer: str
    subject: str
    subject_name: str = ""  # If not provided, will be set to subject
    object_name: str = ""  # If not provided, will be set to answer

    def __post_init__(self) -> None:
        if self.subject_name == "":
            object.__setattr__(self, "subject_name", self.subject)
        if self.object_name == "":
            object.__setattr__(self, "object_name", self.answer)
