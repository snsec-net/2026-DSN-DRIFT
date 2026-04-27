import polars as pl
import gc
from pathlib import Path
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers
)

def get_corpus_batches(file_paths, column="domain", batch_size=10000):
    """
    Scan Parquet files lazily and yield batches for tokenizer training.
    """

    q = pl.scan_parquet(file_paths)
    q = q.select(column).unique().drop_nulls()
    df_unique = q.collect()
    
    total_rows = df_unique.height

    for i in range(0, total_rows, batch_size):
        yield df_unique[column].slice(i, batch_size).to_list()
    
    del df_unique
    gc.collect()

def train(file_paths: list[str],
        text_col: str,
        vocab_size: int,
        min_freq: int,
        save_path: str | Path = "artifacts/tokenizer/tokenizer-{min_freq}-{vocab_size}-both.json",
        use_bert_pretokenizer: bool = False,
    ) -> Tokenizer:

    corpus_iter = get_corpus_batches(file_paths)
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()])

    if use_bert_pretokenizer:
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens)

    tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[PAD]", tokenizer.token_to_id("[PAD]")),
        ],
    )

    tokenizer.save(save_path)