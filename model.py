import torch
from torch import nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx) :
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        torch.nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, input) :
        return self.embedding(input)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        torch.nn.init.xavier_normal_(self.pos_embed.weight)

    def forward(self, x):
        B, L, _ = x.size()
        device = x.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        return x + self.pos_embed(pos_ids)

class Transformer(nn.Module) :
    def __init__(self, d_model, n_heads, dim_feedforward, num_layers, dropout=0.1) :
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)   

    def forward(self, x, mask=None) :
        if mask is not None and mask.any():
            out = self.encoder(x, src_key_padding_mask=mask)
        else:
            out = self.encoder(x)
        return self.dropout(out)
    
class MTPHead(nn.Module) :
    def __init__(self, d_model, vocab_size, dropout=0.1) :
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x) :
        logtis = self.linear(x)
        return logtis
    
class TPPHead(nn.Module) :
    def __init__(self, d_model, vocab_size, dropout=0.1) :
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x) :
        logtis = self.linear(x)
        return logtis
    
class TOVHead(nn.Module):
    def __init__(self, d_model, num_classes=2, dropout=0.1, tov_norm = 'pool'):
        super().__init__()
        self.tov_norm = tov_norm
        if tov_norm == "pool" :
            self.classifier = nn.Linear(d_model * 2, num_classes)
        else :
            self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, sequence_output, padding_mask=None):

        if self.tov_norm == "pool" :
            if padding_mask is not None :
                valid_mask = (~padding_mask).float().unsqueeze(-1)
                sum_embeddings = torch.sum(sequence_output * valid_mask, dim=1)
                sum_mask = torch.sum(valid_mask, dim=1).clamp(min=1)
                avg_output = sum_embeddings / sum_mask

                masked_sequence = sequence_output.masked_fill(padding_mask.unsqueeze(-1), -1e9)
                max_output = masked_sequence.max(dim=1).values
                output = torch.cat((max_output, avg_output), dim=1) 
            else :
                output = torch.cat((sequence_output.max(dim=1).values, sequence_output.mean(dim=1)), dim=1)
        else :
            output = sequence_output[:, 0, :]
        
        logits = self.classifier(output)
        return logits
    
class PretrainedModel(nn.Module) :
    def __init__(self, vocab_size, d_model, n_heads, dim_feedforward, 
                 num_layers, max_len, dropout=0.1, padding_idx=0, tov_norm='pool') :
        super().__init__()

        self.d_model = d_model
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.tov_norm = tov_norm

        self.embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.transformer = Transformer(d_model, n_heads, dim_feedforward, num_layers, dropout)

        self.mtp_head = MTPHead(d_model, vocab_size)

        self.tpp_head = TPPHead(d_model, vocab_size)

        self.tov_head = TOVHead(d_model, num_classes=2, dropout=dropout, tov_norm=self.tov_norm)

    def create_padding_mask(self, input_ids):
        return (input_ids == self.padding_idx)
    
    def forward(self, input_ids, task_type='ALL'):
        token_embed = self.embedding(input_ids)
        x = self.positional_encoding(token_embed)
        padding_mask = self.create_padding_mask(input_ids)
        encoder_output = self.transformer(x, mask=padding_mask)

        valid_mask = (~padding_mask).float().unsqueeze(-1)
        encoder_output = encoder_output * valid_mask

        outputs = {}

        # Task 1: MTP
        if task_type == 'MTP' or task_type == 'ALL':
            outputs['mtp_logits'] = self.mtp_head(encoder_output)
            
        # Task 2: TTP
        if task_type == 'TPP' or task_type == 'ALL':
            outputs['ttp_logits'] = self.tpp_head(encoder_output)
            
        # Task 3: TOV
        if task_type == 'TOV' or task_type == 'ALL':
            outputs['tov_logits'] = self.tov_head(encoder_output, padding_mask=padding_mask)

        # if only one task is required, return only the logits
        if len(outputs) == 1 and task_type != 'ALL':
            return list(outputs.values())[0]
            
        return outputs


class FinetuningHead(nn.Module) :
    def __init__(self, input_dim, d_model, dropout) :
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.dense1 = nn.Linear(input_dim, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model * 2, 2)

    def forward(self, encoder_output) :

        x = self.dropout(encoder_output)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        logits = self.classifier(x)
        return logits
    
class FineTuningModel(nn.Module):
    def __init__(self, pretrain_model_t=None, pretrain_model_c=None,
                 dropout=0.1, padding_idx=0, clf_norm='pool', freeze_backbone=False):
        super().__init__()

        self.padding_idx = padding_idx
        self.clf_norm = clf_norm
        self.use_token = pretrain_model_t is not None
        self.use_char = pretrain_model_c is not None

        sample_model = pretrain_model_t if self.use_token else pretrain_model_c # Extract d_model value
        d_model = sample_model.d_model

        num_active_paths = sum([self.use_token, self.use_char])
        dim_per_path = d_model * 2 if self.clf_norm == 'pool' else d_model
        total_input_dim = dim_per_path * num_active_paths

        # --- Token Path Components ---
        if self.use_token:
            self.transformer_encoder_t = pretrain_model_t.transformer
            self.embedding_t = pretrain_model_t.embedding
            self.positional_encoding_t = pretrain_model_t.positional_encoding
            if freeze_backbone:
                self._set_grad(self.transformer_encoder_t, False)
                self._set_grad(self.embedding_t, False)
                self._set_grad(self.positional_encoding_t, False)

        # --- Character Path Components ---
        if self.use_char:
            self.transformer_encoder_c = pretrain_model_c.transformer
            self.embedding_c = pretrain_model_c.embedding
            self.positional_encoding_c = pretrain_model_c.positional_encoding
            if freeze_backbone:
                self._set_grad(self.transformer_encoder_c, False)
                self._set_grad(self.embedding_c, False)
                self._set_grad(self.positional_encoding_c, False)
        
        # DGA classification head
        self.classifier_head = FinetuningHead(
            input_dim=total_input_dim,
            d_model=d_model,
            dropout=dropout
        )

    def _set_grad(self, module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def create_padding_mask(self, input_ids):
        return (input_ids == self.padding_idx).to(input_ids.device)

    def forward(self, input_ids_t=None, input_ids_c=None):
        features = []

        # --- 1. Token Path (X_t) processing ---
        if self.use_token and input_ids_t is not None:
            t_embed = self.embedding_t(input_ids_t)
            t_x = self.positional_encoding_t(t_embed)
            t_mask = self.create_padding_mask(input_ids_t)
            t_out = self.transformer_encoder_t(t_x, mask=t_mask)

            if self.clf_norm == 'pool':
                # Max pool + Mean pool (d_model * 2)
                valid_mask = (~t_mask).float().unsqueeze(-1)

                t_sum = (t_out * valid_mask).sum(dim=1)
                t_len = valid_mask.sum(dim=1).clamp(min=1)
                t_mean = t_sum / t_len

                t_max = (t_out.masked_fill(valid_mask == 0, -1e9)).max(dim=1).values

                t_feat = torch.cat([t_max, t_mean], dim=1)
            else:
                # CLS Token (d_model * 1)
                t_feat = t_out[:, 0, :]
            features.append(t_feat)

        # --- 2. Character Path (X_c) processing ---
        if self.use_char and input_ids_c is not None:
            c_embed = self.embedding_c(input_ids_c)
            c_x = self.positional_encoding_c(c_embed)
            c_mask = self.create_padding_mask(input_ids_c)
            c_out = self.transformer_encoder_c(c_x, mask=c_mask)

            if self.clf_norm == 'pool':
                # Max pool + Mean pool (d_model * 2)
                valid_mask = (~c_mask).float().unsqueeze(-1)

                c_sum = (c_out * valid_mask).sum(dim=1)
                c_len = valid_mask.sum(dim=1).clamp(min=1)
                c_mean = c_sum / c_len

                c_max = (c_out.masked_fill(valid_mask == 0, -1e9)).max(dim=1).values

                c_feat = torch.cat([c_max, c_mean], dim=1)
            else:
                # CLS Token (d_model * 1)
                c_feat = c_out[:, 0, :]
            features.append(c_feat)

        combined_output = torch.cat(features, dim=1) if len(features) > 1 else features[0]

        return self.classifier_head(combined_output)
    
    def set_backbone_freezing(self, freeze=True):
        """Set the training status of the backbone."""
        trainable = not freeze
        
        if self.use_token:
            for p in self.transformer_encoder_t.parameters(): p.requires_grad = trainable
            for p in self.embedding_t.parameters(): p.requires_grad = trainable
            for p in self.positional_encoding_t.parameters(): p.requires_grad = trainable

        if self.use_char:
            for p in self.transformer_encoder_c.parameters(): p.requires_grad = trainable
            for p in self.embedding_c.parameters(): p.requires_grad = trainable
            for p in self.positional_encoding_c.parameters(): p.requires_grad = trainable
            
        status = "Frozen" if freeze else "Unfrozen"
        print(f"--- Backbone is {status}. ---")