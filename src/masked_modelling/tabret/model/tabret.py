"""TabRet model."""

import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.ft_transformer import FeatureTokenizer, Transformer
from model.rtdl_modules import _TokenInitialization
from model.utils import get_parameter_names


class TabRet(nn.Module):
    """TabRet model."""

    def __init__(
        self,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        feature_tokenizer: FeatureTokenizer,
        encoder: Transformer,
        decoder: Transformer,
        continuous_columns: Optional[List[str]],
        cat_cardinality_dict: Optional[Dict[str, int]],
        initialization: str = "uniform",
    ):
        """Initialize the TabRet model."""
        super().__init__()
        self.decoder_embed_dim = decoder_embed_dim
        self.keys = continuous_columns + list(
            cat_cardinality_dict.keys()  # type: ignore
        )

        self.feature_tokenizer = feature_tokenizer
        self.ft_norm = nn.LayerNorm(encoder_embed_dim)
        self.alignment_layer = nn.Linear(encoder_embed_dim, encoder_embed_dim, bias=True)
        self.encoder = encoder
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(Tensor(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.ParameterDict()
        if continuous_columns is not None:
            self.decoder_pos_embed.update(
                {key: nn.Parameter(Tensor(1, 1, decoder_embed_dim)) for key in continuous_columns}
            )
        if cat_cardinality_dict is not None:
            self.decoder_pos_embed.update(
                {key: nn.Parameter(Tensor(1, 1, decoder_embed_dim)) for key in cat_cardinality_dict}
            )
        self.decoder = decoder

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.projectors = nn.ModuleDict()
        if continuous_columns is not None:
            self.projectors.update({key: nn.Linear(decoder_embed_dim, 1) for key in continuous_columns})
        if cat_cardinality_dict is not None:
            self.projectors.update(
                nn.ModuleDict(
                    {key: nn.Linear(decoder_embed_dim, out_dim) for key, out_dim in cat_cardinality_dict.items()}
                )
            )

        # initialization
        self.initialization_ = _TokenInitialization(initialization)
        self.initialization_.apply(self.mask_token, decoder_embed_dim)
        for parameter in self.decoder_pos_embed.values():
            self.initialization_.apply(parameter, decoder_embed_dim)

    @classmethod
    def make(
        cls,
        continuous_columns: Optional[List[str]],
        cat_cardinality_dict: Optional[Dict[str, int]],
        enc_transformer_config: Dict[str, Any],
        dec_transformer_config: Dict[str, Any],
    ) -> "TabRet":
        """Make the TabRet model."""
        feature_tokenizer = FeatureTokenizer(
            continuous_columns=continuous_columns,  # type: ignore
            cat_cardinality_dict=cat_cardinality_dict,
            d_token=enc_transformer_config["d_token"],
        )

        encoder = Transformer(**enc_transformer_config)
        decoder = Transformer(**dec_transformer_config)

        return TabRet(
            encoder_embed_dim=enc_transformer_config["d_token"],
            decoder_embed_dim=dec_transformer_config["d_token"],
            feature_tokenizer=feature_tokenizer,
            encoder=encoder,
            decoder=decoder,
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
        )

    def optimization_param_groups(self) -> List[Dict[str, Any]]:
        """Return the optimization parameter groups."""
        no_wd_names = ["feature_tokenizer", "normalization", "_norm", ".bias"]
        assert isinstance(getattr(self, no_wd_names[0], None), FeatureTokenizer)
        assert sum(1 for name, _ in self.named_modules() if no_wd_names[1] in name) == (
            len(self.encoder.blocks) + len(self.decoder.blocks)
        ) * 2 - int("attention_normalization" not in self.encoder.blocks[0]) - int(  # type: ignore
            "attention_normalization" not in self.decoder.blocks[0]
        )

        non_decay_parameters = get_parameter_names(self, [nn.LayerNorm, FeatureTokenizer])
        decay_parameters = [name for name in non_decay_parameters if ".bias" not in name]
        return [
            {"params": [p for n, p in self.named_parameters() if n in decay_parameters]},
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

    def freeze_parameters_wo_specific_columns(self, columns: List[str]) -> None:
        """Freeze all the parameters except the specific columns."""
        for name, p in self.named_parameters():
            name_split = name.split(".")
            if len(name_split) > 1 and (name_split[-1] in columns or name_split[-2] in columns):
                p.requires_grad = True
                continue
            p.requires_grad = False

    def freeze_transformer(self) -> None:
        """Freeze the transformer."""
        for p in self.alignment_layer.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder_embed.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze_decoder(self) -> None:
        """Unfreeze the decoder."""
        for p in self.decoder.parameters():
            p.requires_grad = True

    def freeze_parameters(self) -> None:
        """Freeze all the parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """Unfreeze all the parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_mask_token(self) -> None:
        """Unfreeze the mask token."""
        self.mask_token.requires_grad = True

    def show_trainable_parameter(self) -> None:
        """Show the trainable parameters."""
        trainable_list = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_list.append(name)
        # trainable = ", ".join(trainable_list)
        # logger.info(f"Trainable parameters: {trainable_list}")

    def show_frozen_parameter(self) -> None:
        """Show the frozen parameters."""
        frozen_list = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                frozen_list.append(name)
        # frozen = ", ".join(frozen_list)
        # logger.info(f"Frozen parameters: {frozen_list}")

    def add_attribute(
        self,
        continuous_columns: Optional[List[str]] = None,
        cat_cardinality_dict: Optional[Dict[str, int]] = None,
    ) -> None:
        """Add new attributes to the model."""
        assert continuous_columns is not None or cat_cardinality_dict is not None, (
            "At least one of n_num and cardinalities must be presented"
        )
        self.feature_tokenizer.add_attribute(
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
        )

        # add decoder pos embedding
        if continuous_columns is not None:
            for key in continuous_columns:
                if key in self.decoder_pos_embed:
                    continue
                pos_embed_add = nn.Parameter(Tensor(1, 1, self.decoder_embed_dim))
                self.initialization_.apply(pos_embed_add, self.decoder_embed_dim)
                self.decoder_pos_embed.update({key: pos_embed_add})

                self.projectors.update({key: nn.Linear(self.decoder_embed_dim, 1)})

        if cat_cardinality_dict is not None:
            for key, cardinality in cat_cardinality_dict.items():
                if key in self.decoder_pos_embed:
                    continue
                pos_embed_add = nn.Parameter(Tensor(1, 1, self.decoder_embed_dim))
                self.initialization_.apply(pos_embed_add, self.decoder_embed_dim)
                self.decoder_pos_embed.update({key: pos_embed_add})

                self.projectors.update({key: nn.Linear(self.decoder_embed_dim, cardinality)})

    def save_attention_map(
        self,
        x: Tensor,
        keys: Optional[List[str]] = None,
    ) -> None:
        """Save the attention map."""
        if keys is None:
            keys = self.keys

        import os

        os.makedirs("attention/")
        layer = list(range(len(self.encoder.blocks))) + ["all"]
        for i in layer:
            attention = self.encoder.get_attention(x, layer_idx=i)
            print(attention.sum(-1).mean())
            import pandas as pd

            df = pd.DataFrame(attention.mean(0).cpu().numpy(), index=keys, columns=keys)
            df.to_csv(f"./attention/attention_{i}.csv")
        sys.exit()

    def random_masking(
        self,
        x: Tensor,
        mask_ratio: Union[int, float],
        shuffle_idx_shift: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Mask the columns."""
        N, L, D = x.shape

        len_keep = L - mask_ratio if mask_ratio >= 1 else int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if shuffle_idx_shift is not None:
            assert len_keep - len(shuffle_idx_shift) > 0
            noise[:, shuffle_idx_shift[:, 0]] = 0

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def custom_masking(
        self,
        x: Tensor,
        mask_idx: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Mask the columns."""
        N, L, D = x.shape

        mask = torch.zeros(N, L, device=x.device)
        mask[:, mask_idx] = 1

        x_masked = x[:, (1 - mask[0]).bool()]

        ids_restore = torch.ones(N, L, device=x.device) * torch.arange(L, device=x.device)
        return x_masked, mask, ids_restore.to(torch.int64)

    def column_shuffle(self, x: Tensor, column_shuffle_ratio: float, mask: Optional[List[int]] = None) -> Tensor:
        """Shuffle the columns."""
        N, L, _ = x.shape
        num_noise = int(L * column_shuffle_ratio)

        noise = torch.rand(L, device=x.device)  # noise in [0, 1]
        if mask is not None:
            assert num_noise + len(mask) < L
            noise[mask] = 1
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove

        shuffle_idx_shift = torch.cat([ids_shuffle[:, None], torch.randint(N, (L, 1), device=x.device)], dim=1)[
            :num_noise
        ]

        for idx, shift in shuffle_idx_shift:
            x[:, idx] = x[:, idx].roll(shift.item(), 0)
        return x, shuffle_idx_shift

    def forward_encoder(
        self,
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
        mask_ratio: Union[float, List[int]],
        col_shuffle: Optional[Dict[str, Union[int, bool]]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass for the encoder."""
        x = self.feature_tokenizer(x_num, x_cat)
        x = self.ft_norm(x)
        x = self.alignment_layer(x)
        # self.save_attention_map(x, list(x_num.keys()) + list(x_cat.keys()))

        if col_shuffle is not None and col_shuffle["ratio"] > 0:
            x, shuffle_idx_shift = self.column_shuffle(
                x,
                col_shuffle["ratio"],
                mask=mask_ratio if type(mask_ratio) is list else None,
            )
        else:
            shuffle_idx_shift = None
        # masking: length -> length * mask_ratio
        if type(mask_ratio) is not list:
            x, mask, ids_restore = self.random_masking(  # type: ignore
                x,
                mask_ratio,
                shuffle_idx_shift,  # type: ignore
            )
        else:
            x, mask, ids_restore = self.custom_masking(x, mask_ratio)

        x = self.encoder(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore, shuffle_idx_shift

    def forward_decoder(
        self,
        x: Tensor,
        ids_restore: Tensor,
        keys: List[str],
        shuffle_idx_shift: Optional[Tensor] = None,
        col_shuffle: Optional[Dict[str, Union[int, bool]]] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass for the decoder."""
        # embed tokens
        x = self.decoder_embed(x)
        mask_token = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_token], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        if shuffle_idx_shift is not None:
            _, _, d = x.shape
            for idx, shift in shuffle_idx_shift:
                if col_shuffle["mode"] == "concat":  # type: ignore
                    x[:, idx, : d // 2] = x[:, idx, : d // 2].roll(-shift.item(), 0)
                elif col_shuffle["mode"] == "unshuffle":  # type: ignore
                    x[:, idx] = x[:, idx].roll(-shift.item(), 0)
                elif col_shuffle["mode"] == "shuffle":  # type: ignore
                    break

        for i, key in enumerate(keys):
            x[:, i] = x[:, i] + self.decoder_pos_embed[key]

        x = self.decoder(x)
        x = self.decoder_norm(x)

        x_col = {}
        for i, key in enumerate(keys):
            x_col[key] = self.projectors[key](x[:, i])
        return x_col

    def forward_loss(
        self,
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
        pred: Dict[str, Tensor],
        mask: Tensor,
    ) -> Tensor:
        """Compute the loss."""
        # x_num: Dict[L_num, N]
        # x_cat: Dict[L_cat, N]
        # preds Prediction list for each attribute
        # mask: [N, L], 0 is keep, 1 is remove
        if x_num is not None:
            n_num = len(x_num)
        else:
            n_num = 0
            x_num = {}
        if x_cat is None:
            x_cat = {}

        all_loss = 0
        for i, (key, x) in enumerate(x_num.items()):
            loss = F.mse_loss(pred[key].squeeze(), x, reduction="none")
            all_loss += (loss * mask[:, i]).sum()

        for i, (key, x) in enumerate(x_cat.items()):
            loss = F.cross_entropy(pred[key], x, reduction="none")
            all_loss += (loss * mask[:, i + n_num]).sum()

        all_loss /= mask.sum()
        return all_loss

    def forward(
        self,
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
        mask_ratio: Union[float, List[int]],
        col_shuffle: Optional[Dict[str, Union[int, bool]]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """Forward pass."""
        keys = []
        if x_num is not None:
            keys += list(x_num.keys())
        if x_cat is not None:
            keys += list(x_cat.keys())
        x, mask, ids_restore, shuffle_idx_shift = self.forward_encoder(
            x_num,
            x_cat,
            mask_ratio,
            col_shuffle,  # type: ignore
        )
        preds = self.forward_decoder(x, ids_restore, keys, shuffle_idx_shift, col_shuffle)
        loss = self.forward_loss(x_num, x_cat, preds, mask)
        return loss, preds, mask
