from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class BenchmarkConfig:
    repository_root: Path
    config_path: Path
    source_dir: Path
    station_files: dict[str, str]
    source_sha256: dict[str, str]
    targets: tuple[str, ...]
    meteorological_features: tuple[str, ...]
    history_months: int
    horizon_months: int
    train_end: str
    validation_end: str
    test_end: str
    random_seed: int
    ridge_alphas: tuple[float, ...]
    hgb_learning_rates: tuple[float, ...]
    hgb_max_iters: tuple[int, ...]
    hgb_max_leaf_nodes: tuple[int, ...]
    hgb_min_samples_leaf: tuple[int, ...]
    hgb_l2_regularization: tuple[float, ...]
    bootstrap_resamples: int
    bootstrap_block_months: int

    @property
    def config_sha256(self) -> str:
        return sha256(self.config_path.read_bytes()).hexdigest()


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path).expanduser().resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    repository_root = config_path.parent.parent

    data = raw["data"]
    experiment = raw["experiment"]
    models = raw["models"]
    uncertainty = raw["uncertainty"]

    config = BenchmarkConfig(
        repository_root=repository_root,
        config_path=config_path,
        source_dir=(repository_root / data["source_dir"]).resolve(),
        station_files=dict(data["station_files"]),
        source_sha256=dict(data["source_sha256"]),
        targets=tuple(experiment["targets"]),
        meteorological_features=tuple(experiment["meteorological_features"]),
        history_months=int(experiment["history_months"]),
        horizon_months=int(experiment["horizon_months"]),
        train_end=str(experiment["train_end"]),
        validation_end=str(experiment["validation_end"]),
        test_end=str(experiment["test_end"]),
        random_seed=int(experiment["random_seed"]),
        ridge_alphas=tuple(float(value) for value in models["ridge_alphas"]),
        hgb_learning_rates=tuple(float(value) for value in models["hgb_learning_rates"]),
        hgb_max_iters=tuple(int(value) for value in models["hgb_max_iters"]),
        hgb_max_leaf_nodes=tuple(int(value) for value in models["hgb_max_leaf_nodes"]),
        hgb_min_samples_leaf=tuple(int(value) for value in models["hgb_min_samples_leaf"]),
        hgb_l2_regularization=tuple(float(value) for value in models["hgb_l2_regularization"]),
        bootstrap_resamples=int(uncertainty["resamples"]),
        bootstrap_block_months=int(uncertainty["block_months"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: BenchmarkConfig) -> None:
    allowed_targets = {"spei1", "spei3", "spei6", "spei9", "spei12"}
    if not config.targets or not set(config.targets) <= allowed_targets:
        raise ValueError(f"Unsupported targets: {config.targets}")
    if config.history_months < 12:
        raise ValueError("history_months must be at least 12")
    if config.horizon_months != 1:
        raise ValueError("Benchmark v1 supports exactly a one-month horizon")
    if not config.station_files:
        raise ValueError("At least one station file is required")
    if set(config.source_sha256) != set(config.station_files):
        raise ValueError("source_sha256 must bind every configured station exactly once")
    if any(
        len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
        for digest in config.source_sha256.values()
    ):
        raise ValueError("source_sha256 values must be lowercase SHA-256 digests")
    if not config.ridge_alphas or any(alpha < 0 for alpha in config.ridge_alphas):
        raise ValueError("ridge_alphas must contain non-negative values")
    if config.bootstrap_resamples < 1 or config.bootstrap_block_months < 1:
        raise ValueError("Bootstrap settings must be positive")
