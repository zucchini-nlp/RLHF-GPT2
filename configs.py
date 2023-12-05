from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(
        default="gpt2-medium",
        metadata={
            "help": (
                "The model checkpoint for weights initialization."
            )
        },
    )
    reward_model_path: str = field(
        default="reward_model",
        metadata={
            "help": (
                "The model checkpoint for the reward model, either from HF hub or local"
            )
        },
    )
    sft_model: str = field(
        default="gpt_tuned_sft",
        metadata={
            "help": (
                "The model checkpoint for the SFT trained model, either from HF hub or local"
            )
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."},
    )
    load_in_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables BitsAndBytes quantization to load in 4 bits."},
    )
    use_peft: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables PEFT with Lora config."},
    )


@dataclass
class TrainArguments:   
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate or not to evaluate"},
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={
            "help": "Gradient clipping max norm"
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    output_dir: str = field(
        default="./gpt_tuned",
        metadata={"help": "The output dir for logs and checkpoints"},
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={
            "help": "The eval batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    max_steps: int = field(
        default=10000, metadata={"help": "How many optimizer update steps to take"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )
    learning_rate: float = field(default=1e-05, metadata={"help": "The learnign rate"})
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "Learning rate schedule, e.g. constant, linear, cosine."
        },
    )
    warmup_ratio: float = field(
        default=0.10, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    evaluation_strategy: str = field(
        default="no", metadata={"help": "Evaluation strategy, defaults to no eval"}
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=200, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=2,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "Resume training from the last checkpoint"})
    push_to_hub: bool = field(default=False, metadata={"help": "If you want to push the trained model to hub"})
    hub_model_id: str = field(default=None, metadata={"help": "Model id if push_to_hub set to True"})
    report_to: str = field(default="wandb", metadata={"help": "Report logs to wandb, tensorboard or None"})
    local_rank: int = field(default=-1, metadata={"help": "Rank of the process during distributed training."})
    ddp_backend: str = field(default=None, metadata={"help": "The backend to use for distributed training"})


@dataclass
class DatasetArgs(object):    
    dataset_name: str = field(
        metadata={"help": "Dataset id from HF hub."},
    )    
    load_from_local: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether it is a local dataset."},
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset format if load from local (csv, json, pyarrow)."},
    )
    eval_dataset_size: Optional[float] = field(
        default=0.0, metadata={"help": "Size of validation dataset."}
    )
    max_seq_len: int = field(
        default=860,
        metadata={
            "help": "Maximum sequence length. Sequences will be padded (and possibly truncated)."
        },
    )


@dataclass
class PPOArguments:
    learning_rate: float = field(
        default=1e-05,
        metadata={
            "help": "Learning rate."
        },
    )
    log_with: str = field(
        default="wandb",
        metadata={
            "help": "Log to wand or tensorboard"
        },
    )
    mini_batch_size: int = field(
        default=16,
        metadata={
            "help": "Mini batch size."
        },
    )
    batch_size: int = field(
        default=128,
        metadata={
            "help": "Batch size."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Gradient accumulation steps."
        },
    )
    target_kl: float = field(
        default=2.0,
        metadata={
            "help": "Target KL."
        },
    )
    kl_penalty: str = field(
        default="kl",
        metadata={
            "help": "KL penalty."
        },
    )
    steps: int = field(
        default=10000,
        metadata={
            "help": "Steps to train the model."
        },
    )
    early_stopping: bool = field(
        default=False,
        metadata={
            "help": "Enable early stoppping."
        },
    )
    use_score_scaling: bool = field(
        default=True,
        metadata={
            "help": "Use score scaling."
        },
    )
    use_score_norm: bool = field(
        default=True,
        metadata={
            "help": "Use score norm."
        },
    )
    score_clip: float = field(
        default=0.5,
        metadata={
            "help": "Score clip."
        },
    )
    ratio_threshold: float = field(
        default=15.0,
        metadata={
            "help": "Skip mini-batches with high PPO ratios that can cause loss spikes."
        },
    )


@dataclass
class GenerationArguments:
    min_length: int = field(
        default=-1,
        metadata={
            "help": "Path to of model id of the model to be tuned"
        },
    )
    top_k: float = field(
        default=0.0,
        metadata={
            "help": "Path to or model id of the reward model."
        },
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Path to or model id of the reward model."
        },
    )
    do_sample: bool = field(
        default=True,
        metadata={
            "help": "Path to or model id of the reward model."
        },
    )
    max_new_tokens: int = field(
        default=512,
        metadata={
            "help": "Path to or model id of the reward model."
        },
    )
