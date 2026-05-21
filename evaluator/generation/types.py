from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenerationSample:
    uid: str
    benchmark: str
    prompt: str
    stage: str = "generate"
    expected_outputs: int = 1
    task: str = ""
    media: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    answer: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    uid: str
    benchmark: str
    stage: str = "generate"
    status: str = "pending"
    request_path: str = ""
    response_path: str = ""
    output_dir: str = ""
    text_answer: str = ""
    image_paths: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    benchmark: str
    display_name: str
    model: str
    run_id: str
    phase: str
    backend: str
    n_samples: int
    main_metric_name: str = ""
    main_metric_value: float | None = None
    summary_path: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

