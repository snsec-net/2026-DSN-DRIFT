import polars as pl
pl.Config.set_engine_affinity(engine="streaming")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import FineTuningDataset
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from model import PretrainedModel, FineTuningModel
from tqdm import tqdm
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
from utility import dataset
from utility.path import path_tokenizer, path_model
from utility.config import FinetuningConfig
import argparse

def load_pretrain_weights(pt_model, weigths_path, device) :
    state_dict = torch.load(weigths_path, map_location=device)

    weights_to_load = {}

    for name, param in state_dict.items() :
        if name.startswith('transformer.') or name.startswith('embedding.') or name.startswith('positional_encoding.') :
            weights_to_load[name] = param

    pt_model.load_state_dict(weights_to_load, strict=False)
    return pt_model

def fine_tune_dga_classifier(pt_model_t, pt_model_c, train_dataloader, val_dataloader, device, save_path, args):

    pt_t = load_pretrain_weights(pt_model_t, args.token_weights_path, device) if args.use_token else None
    pt_c = load_pretrain_weights(pt_model_c, args.char_weights_path, device) if args.use_char else None
    
    ft_model = FineTuningModel(
        pretrain_model_t=pt_t, 
        pretrain_model_c=pt_c, 
        freeze_backbone=args.freeze_backbone,
        clf_norm=args.clf_norm
    ).to(device)

    param_groups = [
        {
            'params': ft_model.classifier_head.parameters(), 
            'lr': args.learning_rate
        }
    ]
    
    if ft_model.use_token:
        param_groups.extend([
            {'params': p.parameters(), 'lr': args.backbone_lr} 
            for p in [ft_model.transformer_encoder_t, ft_model.embedding_t, ft_model.positional_encoding_t]
        ])
    if ft_model.use_char:
        param_groups.extend([
            {'params': p.parameters(), 'lr': args.backbone_lr} 
            for p in [ft_model.transformer_encoder_c, ft_model.embedding_c, ft_model.positional_encoding_c]
        ])

    optimizer = optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()

    if args.freeze_backbone:
        unfreeze_step = int(len(train_dataloader) * args.unfreeze_at_epoch) if args.unfreeze_at_epoch is not None else float('inf')
        backbone_unfrozen = False
    else: # full finetuning
        unfreeze_step = None
        backbone_unfrozen = True

    best_val_loss = float('inf')
    global_step = 0    
    interval_loss_sum_total = 0
    interval_batch_counter = 0

    for epoch in range(args.num_epochs) :
        ft_model.train()
        total_loss = 0
        train_loop = tqdm(train_dataloader, desc=f'FineTune Epoch {epoch+1}', 
                        bar_format="{l_bar}{n_fmt}/{total_fmt} | [{elapsed}<{remaining} {postfix}]",
                        leave=False)

        for X_token, X_char, y_train in train_loop :
            global_step += 1

            if args.freeze_backbone and not backbone_unfrozen and global_step >= unfreeze_step:
                ft_model.set_backbone_freezing(freeze=False)
                backbone_unfrozen = True
                print(f"--- [Step {global_step}] Backbone Unfrozen. Momentum Preserved. ---")

            X_token, X_char, y_train = X_token.to(device), X_char.to(device), y_train.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.use_bf16):
                logits = ft_model(
                    X_token if args.use_token else None, 
                    X_char if args.use_char else None
                )
                loss = criterion(logits, y_train)
            
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']

            total_loss += loss.item()

            interval_loss_sum_total += loss.item()
            interval_batch_counter += 1

            if interval_batch_counter == args.log_interval_steps :

                avg_total_interval_loss = interval_loss_sum_total / interval_batch_counter


                if global_step % args.log_interval_steps == 0 :

                    avg_val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_finetuning(ft_model, val_dataloader, device, args.use_bf16)

                    wandb.log({
                        'train/loss' : avg_total_interval_loss,
                        'val/loss': avg_val_loss,
                        'val/acc' : val_acc,
                        'val/prec': val_precision,
                        'val/recall' : val_recall,
                        'val/f1': val_f1,
                        'train/lr': current_lr
                    }, step=global_step)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(ft_model.state_dict(), save_path)

                    train_loop.write(f"[Step {global_step} Interval Log]: Train Loss: {avg_total_interval_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f},"
                        f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
                else :
                    wandb.log({
                        'train/loss' : avg_total_interval_loss,
                        'train/lr': current_lr
                    }, step=global_step)

                interval_loss_sum_total = 0 
                interval_batch_counter = 0

            current_step = train_loop.n + 1
            avg_total = total_loss / current_step

            train_loop.set_postfix(avg_loss=f'{avg_total:.4f}', refresh=False)

def evaluate_finetuning(model, dataloader, device, use_bf16):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    val_preds = []
    val_labels = []

    val_loop = tqdm(dataloader, desc="Validation", disable=True,
                    bar_format="{l_bar}{n_fmt}/{total_fmt} | [{elapsed}<{remaining} {postfix}]",
                    leave=False)

    with torch.no_grad():
        for X_token, X_char, y_val in val_loop:
            X_token, X_char, y_val = X_token.to(device), X_char.to(device), y_val.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(
                    X_token if model.use_token else None, 
                    X_char if model.use_char else None
                )
                loss = criterion(logits, y_val)
            
            total_loss += loss.item()

            predicted_labels = torch.argmax(logits, dim=1)

            val_preds.append(predicted_labels.cpu())
            val_labels.append(y_val.cpu())

            current_step = val_loop.n + 1
            avg_total = total_loss / current_step

            val_loop.set_postfix(avg_loss=f'{avg_total:.4f}', refresh=False)

    avg_loss = total_loss / len(dataloader)

    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)

    accuracy = (val_preds == val_labels).sum().item() / len(val_labels)
    precision = precision_score(val_labels, val_preds, zero_division=0)
    recall = recall_score(val_labels, val_preds, zero_division=0)
    f1 = f1_score(val_labels, val_preds, zero_division=0)

    model.train()
    return avg_loss, accuracy, precision, recall, f1

def main():
    cfg = FinetuningConfig()

    parser = argparse.ArgumentParser(description="Fine-tuning DGA Classifier")

    # path
    parser.add_argument("--tokenizer_path", type=str, default=cfg.tokenizer_path)
    parser.add_argument("--use_bert_pretokenizer", type=bool, default=False)
    parser.add_argument("--project_name", type=str, default=cfg.project_name)
    parser.add_argument("--best_filename", type=str, default=cfg.best_filename)
    parser.add_argument("--wandb_mode", type=str, default=cfg.wandb_mode)
    parser.add_argument("--timestamp", type=str, default=cfg.timestamp)

    # model
    parser.add_argument("--d_model", type=int, default=cfg.d_model)
    parser.add_argument("--nhead", type=int, default=cfg.nhead)
    parser.add_argument("--dim_feedforward", type=int, default=cfg.dim_feedforward)
    parser.add_argument("--num_layers", type=int, default=cfg.num_layers)
    parser.add_argument("--max_len_token", type=int, default=cfg.max_len_token)
    parser.add_argument("--max_len_char", type=int, default=cfg.max_len_char)
    parser.add_argument("--vocab_size_char", type=int, default=cfg.vocab_size_char)

    # hyperparameter
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument("--num_epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--learning_rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--backbone_lr", type=float, default=cfg.backbone_lr)
    parser.add_argument("--log_interval_steps", type=int, default=cfg.log_interval_steps)

    # flag
    parser.add_argument("--use_token", default=cfg.use_token)
    parser.add_argument("--use_char", default=cfg.use_char)
    parser.add_argument("--freeze_backbone", default=cfg.freeze_backbone)
    parser.add_argument("--unfreeze_at_epoch", type=float, default=0.5)
    parser.add_argument("--clf_norm", type=str, default=cfg.clf_norm, choices=['cls', 'pool'])
    parser.add_argument("--use_bf16", action="store_true", help="Use bf16")
    
    # weight path
    parser.add_argument("--token_weights_path", type=str, default=cfg.token_weights_path)
    parser.add_argument("--char_weights_path", type=str, default=cfg.char_weights_path)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # weight path
    args.token_weights_path = path_model.joinpath(args.token_weights_path)
    args.char_weights_path = path_model.joinpath(args.char_weights_path)

    # tokenizer & path & wandb
    if args.use_bert_pretokenizer :
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    else :
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path_tokenizer.joinpath(args.tokenizer_path)))
    vocab_size_token = tokenizer.vocab_size

    save_dir = path_model.joinpath(args.timestamp)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir.joinpath(f"{args.best_filename}.pt")

    wandb.init(project=args.project_name, name=args.best_filename, 
               config=vars(args), mode=args.wandb_mode, tags=['valid'])
               
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")

    # dataset
    train_df, _ = dataset.get_train_set()
    val_df = dataset.get_val_set()

    train_dataset = FineTuningDataset(train_df, tokenizer=tokenizer, 
                                      max_len_t=args.max_len_token, max_len_c=args.max_len_char)
    val_dataset = FineTuningDataset(val_df, tokenizer=tokenizer, 
                                    max_len_t=args.max_len_token, max_len_c=args.max_len_char)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=args.num_workers)

    pt_model_t = PretrainedModel(vocab_size=vocab_size_token, d_model=args.d_model, 
                                 n_heads=args.nhead, dim_feedforward=args.dim_feedforward, 
                                 num_layers=args.num_layers, max_len=args.max_len_token)
    pt_model_c = PretrainedModel(vocab_size=args.vocab_size_char, d_model=args.d_model, 
                                 n_heads=args.nhead, dim_feedforward=args.dim_feedforward, 
                                 num_layers=args.num_layers, max_len=args.max_len_char)
    
    # train
    fine_tune_dga_classifier(
        pt_model_t,
        pt_model_c,
        train_dataloader,
        val_dataloader,
        device=device,
        save_path=best_model_path,
        args=args
    )
    
    if best_model_path.exists():
        artifact = wandb.Artifact(name=args.best_filename, type="model")
        artifact.add_file(str(best_model_path))
        wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()