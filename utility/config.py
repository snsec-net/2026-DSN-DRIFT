from dataclasses import dataclass, field
import datetime

@dataclass
class PretrainConfig:
    # Data
    max_len_char: int = 77
    vocab_size_char: int = 43
    text_col: str = "domain"
    label_col: str = "label"
    
    # Vocabulary (Subword)
    max_len_subword: int = 30
    vocab_size_subword: int = 30522
    min_freq_subword: int = 0
    use_bert_pretokenizer: bool = False
    
    # Model
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 12
    dim_feedforward: int = 768
    dropout: float = 0.1
    
    # Training
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 1e-4
    
    # Pretraining Tasks
    mask_ratio: float = 0.15
    shuffle_prob: float = 0.5
    tov_norm: str = "pool"  # "cls" or "pool"
    ignore_index: int = -100
    
    # Output
    save_path: str = "pretrained.pt"


@dataclass
class FinetuningConfig:
    token_weights_path: str = 'SUBWORD_BACKBONE.pt'
    char_weights_path: str = 'CHAR_BACKBONE.pt'
    tokenizer_path: str = "tokenizer-0-30522-both.json"

    d_model: int = 256
    nhead: int = 8
    num_layers: int = 12
    dim_feedforward: int = 768
    max_len_token: int = 30
    max_len_char: int = 77
    vocab_size_token: int = 30522 # tokenizer_m.vocab_size
    vocab_size_char: int = 43

    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    backbone_lr: float = 1e-6
    num_workers: int = 4
    log_interval_steps: int = 1000
    unfreeze_at_epoch: float = 0.5

    # Ablation Study / Training Strategy
    use_token: bool = True       # Token Backbone usage
    use_char: bool = True        # Char Backbone usage
    freeze_backbone: bool = True # Backbone freezing (if True, only Head is trained at the beginning)
    clf_norm: str = "pool"       # "cls" or "pool"

    # Logging & Project
    project_name: str = 'drift-finetune'
    wandb_mode: str = 'online' # 'online', 'offline', 'disabled'
    run_name_prefix: str = 'finetuning'
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime('%m%d_%H%M'))

    @property
    def best_filename(self) -> str:
        return f"{self.run_name_prefix}_{self.timestamp}"