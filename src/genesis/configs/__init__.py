# this file is needed here to include configs when building project as a package
from omegaconf import OmegaConf


def register_config_resolvers() -> None:
    """Register custom OmegaConf resolvers to handle complex config interpolation cases."""
    OmegaConf.register_new_resolver("cfg.graph_level_criterion", lambda task: _graph_level_criterion_resolver(task))


def _graph_level_criterion_resolver(task: str) -> str:
    """Resolver that determines the criterion to use for the model based on the data task."""
    match task:
        case "regression":
            return "torch.nn.L1Loss"
        case "multiclass":
            return "torch.nn.CrossEntropyLoss"
        case "multilabel":
            return "torch.nn.BCEWithLogitsLoss"
        case "binary":
            return "torch.nn.BCEWithLogitsLoss"
        case _:
            raise ValueError(
                f"Unsupported task type: {task}. Supported tasks are: regression, multiclass, multilabel, binary."
            )
