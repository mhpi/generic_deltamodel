import torch
from models.neural_networks.transformer.features_embedding import (
    StaticDecEmbedding,
    StaticEncEmbedding,
    TimeSeriesDecEmbedding,
    TimeSeriesEncEmbedding,
)
from models.neural_networks.transformer.mask import MaskGenerator
from models.neural_networks.transformer.positional_encoding import PositionalEncoding
from models.neural_networks.transformer.transformer_layers import TransformerBackbone
from torch import nn


class Model(nn.Module):
    """
    StefaLand encoder (TransformerBackbone) with a single-layer LSTM
    decoder. This is the architecture StefaLandGlobal20.pt was pretrained
    with, and the checkpoint loads into it with an exact parameter match.
    """

    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        embed_dim = configs.d_model
        self.embed_dim = embed_dim
        d_ffd = configs.d_ffd

        self.time_series_variables = configs.time_series_variables
        self.static_variables = configs.static_variables
        self.static_variables_category = configs.static_variables_category
        self.static_variables_category_num = [
            len(configs.static_variables_category_dict[x]['class_to_index'])
            for x in configs.static_variables_category
        ]
        self.static_variables_numeric = [
            var
            for var in self.static_variables
            if var not in self.static_variables_category
        ]

        # Norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Feature embeddings
        self.time_series_embedding = TimeSeriesEncEmbedding(
            configs.time_series_variables, embed_dim, dropout=configs.dropout
        )
        self.static_embedding = StaticEncEmbedding(
            self.static_variables_numeric,
            embed_dim,
            categorical_features=self.static_variables_category,
            categorical_features_num=self.static_variables_category_num,
            dropout=configs.dropout,
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=configs.dropout)

        # Mask generator
        self.mask_generator = MaskGenerator(
            configs.mask_ratio_time_series,
            mask_ratio_static=configs.mask_ratio_static,
            min_window_size=configs.min_window_size,
            max_window_size=configs.max_window_size,
        )

        # Encoder
        self.encoder = TransformerBackbone(
            embed_dim, configs.num_enc_layers, d_ffd, configs.num_heads, configs.dropout
        )

        # Decoder: single-layer LSTM (not a TransformerBackbone)
        self.enc_2_dec_embedding = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)

        # Projection layers
        self.time_series_projection = TimeSeriesDecEmbedding(
            configs.time_series_variables,
            embed_dim,
            dropout=configs.dropout,
            add_input_noise=configs.add_input_noise,
        )
        self.static_projection = StaticDecEmbedding(
            self.static_variables_numeric,
            embed_dim,
            categorical_features=self.static_variables_category,
            categorical_features_num=self.static_variables_category_num,
            dropout=configs.dropout,
            add_input_noise=configs.add_input_noise,
        )
        self.dropout = nn.Dropout(configs.dropout)
        self.init_weights()

    def init_weights(self):
        def init_layer(layer):
            nn.init.uniform_(
                layer.weight, -self.configs.init_weight, self.configs.init_weight
            )
            nn.init.uniform_(
                layer.bias, -self.configs.init_bias, self.configs.init_bias
            )

        nn.init.uniform_(
            self.positional_encoding.position_embedding,
            -self.configs.init_weight,
            self.configs.init_weight,
        )

        layers_to_init = [
            *self.time_series_embedding.embeddings1.values(),
            *self.time_series_embedding.embeddings2.values(),
            *self.static_embedding.numerical_embeddings1.values(),
            *self.static_embedding.numerical_embeddings2.values(),
            *self.time_series_projection.embeddings1.values(),
            *self.time_series_projection.embeddings2.values(),
            *self.static_projection.numerical_embeddings1.values(),
            *self.static_projection.numerical_embeddings2.values(),
        ]

        for layer in layers_to_init:
            init_layer(layer)

        nn.init.uniform_(
            self.time_series_embedding.masked_values,
            -self.configs.init_weight,
            self.configs.init_weight,
        )
        nn.init.uniform_(
            self.static_embedding.masked_values,
            -self.configs.init_weight,
            self.configs.init_weight,
        )

    def forward(self, batch_data_dict, is_mask=True):
        batch_x = batch_data_dict['batch_x']  # [B, T, F]
        batch_c = batch_data_dict['batch_c']  # [B, C]
        masked_time_series_index = batch_data_dict['batch_time_series_mask_index']
        masked_static_index = batch_data_dict['batch_static_mask_index']

        if is_mask:
            if masked_time_series_index.numel() == 0:
                _, masked_time_series_index = self.mask_generator(
                    batch_x.shape, method="consecutive"
                )
                _, masked_static_index = self.mask_generator(
                    batch_c.shape, method="isolated_point"
                )

            masked_time_series_index = masked_time_series_index.to(batch_x.device)
            masked_static_index = masked_static_index.to(batch_x.device)

            masked_missing_time_series_index = torch.isnan(batch_x)
            masked_missing_static_index = torch.isnan(batch_c)
            batch_x = batch_x.masked_fill(masked_missing_time_series_index, 0)
            batch_c = batch_c.masked_fill(masked_missing_static_index, 0)

            masked_time_series_index = (
                masked_time_series_index | masked_missing_time_series_index
            )
            masked_static_index = masked_static_index | masked_missing_static_index
        else:
            masked_missing_time_series_index = torch.isnan(batch_x)
            masked_missing_static_index = torch.isnan(batch_c)

        enc_x = self.time_series_embedding(
            batch_x,
            feature_order=self.time_series_variables,
            masked_index=masked_time_series_index,
        )
        enc_c = self.static_embedding(
            batch_c,
            feature_order=self.static_variables,
            masked_index=masked_static_index,
        )

        enc_x = torch.cat([enc_x, enc_c[:, None, :]], dim=1)
        enc_x = self.positional_encoding(enc_x)

        enc_bs, enc_seq_len, enc_d_model = enc_x.shape
        if self.configs.warmup_train:
            enc_x = torch.cat([enc_x[:, : int(enc_x.shape[1] / 2), :], enc_x], dim=1)

        hidden_states = self.encoder(enc_x)
        hidden_states = self.encoder_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.enc_2_dec_embedding(hidden_states)

        # The encoder's own latent representation, taken before the decoder.
        # The LSTM decoder below belongs to the masked-reconstruction
        # pretraining head, not to what fine-tuning should consume.
        encoder_hidden = hidden_states[:, -enc_seq_len:, :]
        encoder_hidden_time_series = encoder_hidden[:, :-1, :]  # [B, T, d_model]
        encoder_hidden_static = encoder_hidden[:, -1, :]  # [B, d_model]

        if self.configs.num_dec_layers > 0:
            dec_x, _ = self.decoder(hidden_states)
            dec_x = self.decoder_norm(dec_x)
            dec_x = self.dropout(dec_x)
        else:
            dec_x = hidden_states

        dec_x = dec_x[:, -enc_seq_len:, :]

        dec_x_time_series = dec_x[:, :-1, :]  # [B, T, d_model]
        dec_x_static = dec_x[:, -1, :]  # [B, d_model]

        outputs_time_series = self.time_series_projection(
            dec_x_time_series, feature_order=self.time_series_variables
        )
        (
            outputs_static,
            static_variables_dec_index_start,
            static_variables_dec_index_end,
        ) = self.static_projection(
            dec_x_static,
            feature_order=self.static_variables,
            mode=batch_data_dict['mode'],
        )

        output_dict = {
            'outputs_time_series': outputs_time_series,
            'outputs_static': outputs_static,
            'masked_time_series_index': masked_time_series_index,
            'masked_static_index': masked_static_index,
            'masked_missing_time_series_index': masked_missing_time_series_index,
            'masked_missing_static_index': masked_missing_static_index,
            'static_variables_dec_index_start': static_variables_dec_index_start,
            'static_variables_dec_index_end': static_variables_dec_index_end,
            'encoder_hidden_time_series': encoder_hidden_time_series,
            'encoder_hidden_static': encoder_hidden_static,
        }

        return output_dict
