"""Typed, serializable configuration for clean AlphaZero runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class ArchitectureConfig:
    input_channels: int = 18
    channels: int = 128
    residual_blocks: int = 8
    policy_channels: int = 32
    value_channels: int = 32
    action_space_size: int = 4544


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 400
    batch_size: int = 32
    c_puct: float = 1.5
    max_tree_nodes: int | None = None
    engine: str = "python_gpu_reference"


@dataclass(frozen=True)
class SelfPlayConfig:
    num_games: int = 256
    max_moves: int = 400
    temperature: float = 1.0
    temperature_moves: int = 30
    late_temperature: float = 0.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    workers_per_gpu: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    steps: int = 1_200
    optimizer: str = "adamw"
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    channels_last: bool = True
    scheduler: str = "none"


@dataclass(frozen=True)
class EvaluationConfig:
    games: int = 40
    simulations: int = 400
    temperature: float = 0.0
    max_moves: int = 400
    confidence_level: float = 0.95
    min_completed_games: int = 32
    promotion_score: float = 0.55
    include_truncations: bool = False


@dataclass(frozen=True)
class StorageConfig:
    run_root: str = "runs/clean_az"
    replay_capacity: int = 150_000
    checkpoint_frequency: int = 1
    iteration_checkpoint_retention: int = 5
    replay_retention: int = 1


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    num_gpus: int | None = 2
    deterministic: bool = False
    allow_tf32: bool = True
    cpu_test_mode: bool = False


@dataclass(frozen=True)
class RunConfig:
    """All user-tunable clean-run settings in one object."""

    run_name: str = "clean_az_v1"
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if not self.run_name or any(c in self.run_name for c in "\\/"):
            raise ValueError("run_name must be a non-empty filename component.")
        if self.architecture.input_channels != 18:
            raise ValueError("The audited baseline requires 18 input channels.")
        if self.architecture.action_space_size != 4544:
            raise ValueError("The audited baseline requires the 4544-action space.")
        if self.mcts.simulations < 100:
            raise ValueError("Production MCTS must support at least 100 simulations.")
        if self.mcts.batch_size <= 0:
            raise ValueError("mcts.batch_size must be positive.")
        if self.mcts.engine not in {"python_gpu_reference", "cpp"}:
            raise ValueError("mcts.engine must be 'python_gpu_reference' or 'cpp'.")
        if self.self_play.num_games <= 0 or self.self_play.max_moves <= 0:
            raise ValueError("self-play game and move counts must be positive.")
        if self.self_play.temperature < 0 or self.self_play.late_temperature < 0:
            raise ValueError("temperatures must be non-negative.")
        if self.self_play.temperature_moves < 0:
            raise ValueError("temperature_moves must be non-negative.")
        if self.self_play.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive.")
        if not 0 <= self.self_play.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1].")
        if self.training.batch_size <= 0 or self.training.steps <= 0:
            raise ValueError("training batch size and steps must be positive.")
        if self.training.learning_rate <= 0 or self.training.weight_decay < 0:
            raise ValueError("invalid optimizer settings.")
        if self.training.optimizer not in {"sgd", "adam", "adamw"}:
            raise ValueError("unsupported optimizer.")
        if self.evaluation.games <= 0 or self.evaluation.games % 2:
            raise ValueError("evaluation.games must be a positive, even number.")
        if self.evaluation.simulations < 100:
            raise ValueError("evaluation requires at least 100 simulations.")
        if self.evaluation.temperature < 0:
            raise ValueError("evaluation temperature must be non-negative.")
        if not 0 < self.evaluation.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1).")
        if not 0.5 <= self.evaluation.promotion_score <= 1:
            raise ValueError("promotion_score must be in [0.5, 1].")
        if self.storage.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive.")
        if self.storage.iteration_checkpoint_retention < 1:
            raise ValueError("at least one iteration checkpoint must be retained.")

    @property
    def root(self) -> Path:
        return Path(self.storage.run_root).resolve() / self.run_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        config = cls(
            run_name=raw.get("run_name", cls.run_name),
            architecture=ArchitectureConfig(**raw.get("architecture", {})),
            mcts=MCTSConfig(**raw.get("mcts", {})),
            self_play=SelfPlayConfig(**raw.get("self_play", {})),
            training=TrainingConfig(**raw.get("training", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            storage=StorageConfig(**raw.get("storage", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )
        config.validate()
        return config

    @classmethod
    def load_json(cls, path: str | Path) -> "RunConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
