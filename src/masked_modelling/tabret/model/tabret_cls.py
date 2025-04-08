"""TabRetClassifier class."""

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from transtab.trainer_utils import get_parameter_names

from model.ft_transformer import FeatureTokenizer
from model.tabret import TabRet


class TabRetClassifier(nn.Module):
    """TabRetClassifier class."""

    def __init__(self, pretrained_tabret: TabRet, output_dim: int) -> None:
        """Initialize the TabRetClassifier."""
        super().__init__()

        self.feature_tokenizer = pretrained_tabret.feature_tokenizer
        self.ft_norm = pretrained_tabret.ft_norm
        self.alignment_layer = pretrained_tabret.alignment_layer
        # self.al_norm = pretrained_tabret.al_norm
        self.encoder = pretrained_tabret.encoder
        self.encoder_norm = pretrained_tabret.encoder_norm

        self.decoder_embed = pretrained_tabret.decoder_embed
        self.mask_token = pretrained_tabret.mask_token
        self.decoder_pos_embed = pretrained_tabret.decoder_pos_embed
        self.decoder = pretrained_tabret.decoder
        self.decoder_norm = pretrained_tabret.decoder_norm

        self.pos_embed_target = nn.Parameter(Tensor(1, 1, pretrained_tabret.decoder_embed_dim))
        pretrained_tabret.initialization_.apply(self.pos_embed_target, pretrained_tabret.decoder_embed_dim)

        self.classifier = nn.Linear(pretrained_tabret.decoder_embed_dim, output_dim)
        if output_dim > 1:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def optimization_param_groups(self) -> List[Dict[str, Tensor]]:
        """Return the optimization parameter groups."""
        non_decay_parameters = get_parameter_names(self, [nn.LayerNorm, FeatureTokenizer])
        decay_parameters = [name for name in non_decay_parameters if ".bias" not in name]
        return [
            {"params": [p for n, p in self.named_parameters() if n in decay_parameters]},
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

    def show_trainable_parameter(self) -> None:
        """Show the trainable parameters."""
        trainable_list = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_list.append(name)
        # trainable = ", ".join(trainable_list)

    def forward_encoder(
        self,
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
    ) -> Tensor:
        """Forward pass for the encoder."""
        x = self.feature_tokenizer(x_num, x_cat)
        x = self.ft_norm(x)
        x = self.alignment_layer(x)
        # x = self.al_norm(x)
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x

    def forward_decoder(self, x: Tensor, keys: List[str]) -> Tensor:
        """Forward pass for the decoder."""
        x = self.decoder_embed(x)

        mask_token = self.mask_token.repeat(x.shape[0], 1, 1)
        mask_token = mask_token + self.pos_embed_target

        for i, key in enumerate(keys):
            x[:, i] = x[:, i] + self.decoder_pos_embed[key]

        x = torch.cat([mask_token, x], dim=1)

        x = self.decoder(x)
        x = self.decoder_norm(x)
        return x

    def forward(
        self,
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
    ) -> Tensor:
        """Forward pass."""
        keys = []
        if x_num is not None:
            keys += list(x_num.keys())
        if x_cat is not None:
            keys += list(x_cat.keys())

        x = self.forward_encoder(x_num, x_cat)
        x = self.forward_decoder(x, keys)
        logit = self.classifier(x[:, 0])
        return logit
