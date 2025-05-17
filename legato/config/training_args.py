from dataclasses import dataclass, field

@dataclass
class DataArguments:
    dataset_path: str
    mini_val_file: str = field(
        default=None,
        metadata={"help": "Path to the mini validation file. If not provided, the the whole validation set will be used."},
    )
    mini_test_file: str = field(
        default=None,
        metadata={"help": "Path to the mini test file. If not provided, the the whole test set will be used."},
    )
    dummy_data: bool = field(
        default=False,
        metadata={"help": "Use dummy data (32 items) for testing purposes."},
    )

@dataclass
class ModelArguments:
    model_config: str
    pretrained_model: str = field(
        default=None,
        metadata={"help": "Set the path to load a pretrained model before training or evaluation."},
    )
