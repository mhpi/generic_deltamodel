import torch
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
import math

class TimeSeriesEncEmbedding(nn.Module):
    def __init__(self, time_series_variables, embed_dim, dropout=0.1):
        super().__init__()
        self.time_series_variables = time_series_variables
        self.num_features = len(time_series_variables)
        self.embed_dim = embed_dim

        self.embeddings1 = nn.ModuleDict({name: nn.Linear(1, 64, bias=True) for name in time_series_variables})
        self.embeddings2 = nn.ModuleDict({name: nn.Linear(64, embed_dim, bias=True) for name in time_series_variables})

        self.masked_values = nn.Parameter(torch.randn(self.num_features, embed_dim), requires_grad=True)
        # self.masked_values = nn.Parameter(torch.randn(1000, 500, self.num_features), requires_grad=True)

        self.dropout = nn.Dropout(dropout)

    #     self.init_weights()
    #
    # def init_weights(self):
    #     # truncated normal distribution
    #     trunc_normal_(self.masked_values, std=.02)

    def forward(self, x, feature_order, masked_index=None):
        """
        Input:
            x: batch_size, seq_length, num_features
            feature_order: list of feature names
            masked_index: batch_size, seq_length, num_features
            masked_vector: num_features, embed_dim
        """

        num_bs, seq_len, num_features = x.shape
        # if masked_index is not None:
        #     masked_vector = self.masked_values[:num_bs, :seq_len, :]
        #     masked_index = masked_index.to(masked_vector.device)
        #     x = torch.where(masked_index, masked_vector, x)

        embeds = []
        for i, name in enumerate(feature_order):
            embed1 = self.embeddings1[name](x[..., i:i + 1]) # batch_size, seq_length, 64
            embed1 = F.gelu(embed1)
            embed1 = self.dropout(embed1)
            embed2 = self.embeddings2[name](embed1) # batch_size, seq_length, embed_dim
            embeds.append(embed2)

        embeds = torch.stack(embeds, dim=-2) # batch_size, seq_length, num_features, embed_dim

        if masked_index is not None:
            # --> batch_size, seq_length, num_features, embed_dim
            masked_vector = self.masked_values.unsqueeze(0).unsqueeze(0).expand(num_bs, seq_len, -1, -1)
            # masked_index = masked_index.unsqueeze(-1).expand_as(embeds).to(masked_vector.device)
            masked_index = masked_index.unsqueeze(-1).to(masked_vector.device)
            embeds = torch.where(masked_index, masked_vector, embeds)

        embeds = torch.sum(embeds, dim=-2) # batch_size, seq_length, embed_dim
        # embeds = torch.nansum(embeds, dim=-2) # batch_size, seq_length, embed_dim

        return embeds


class TimeSeriesDecEmbedding(nn.Module):
    def __init__(self, time_series_variables, embed_dim, dropout=0.1, add_input_noise=False):
        super().__init__()
        output_dim = 1
        if add_input_noise:
            output_dim = 2
        self.embeddings1 = nn.ModuleDict({name: nn.Linear(embed_dim, 64, bias=True) for name in time_series_variables})
        self.embeddings2 = nn.ModuleDict({name: nn.Linear(64, output_dim, bias=True) for name in time_series_variables})
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, feature_order):
        embeds = []
        for i, name in enumerate(feature_order):
            embed1 = self.embeddings1[name](x) # batch_size, seq_length, 64
            embed1 = F.gelu(embed1)
            embed1 = self.dropout(embed1)
            embed2 = self.embeddings2[name](embed1) # batch_size, seq_length, 1
            embeds.append(embed2)
        embeds = torch.concatenate(embeds, dim=-1) # batch_size, seq_length, num_features
        return embeds

class TimeSeriesDecEmbeddingLSTM(nn.Module):
    def __init__(self, time_series_variables, embed_dim, dropout=0.4, add_input_noise=False):
        super().__init__()
        output_dim = 1
        if add_input_noise:
            output_dim = 2
        # Replace the first linear layer with LSTM
        self.embeddings1 = nn.ModuleDict(
            {name: nn.LSTM(embed_dim, 64, batch_first=True) for name in time_series_variables})
        # Keep the second linear layer unchanged to transform LSTM's output dimension to 1
        self.embeddings2 = nn.ModuleDict({name: nn.Linear(64, output_dim, bias=True) for name in time_series_variables})
        self.dropout = nn.Dropout(0.4)
        print("Fixed dropout rate of 0.4 for TimeSeriesDecEmbeddingLSTM")

    def forward(self, x, feature_order):
        embeds = []
        for i, name in enumerate(feature_order):
            # LSTM's input should be in the form (batch_size, seq_length, feature_dim)
            # LSTM outputs a tuple, the first element being the output at all time steps, and the second the hidden state at the last time step
            # We are only interested in the output
            lstm_output, _ = self.embeddings1[name](x)  # batch_size, seq_length, hidden_dim
            lstm_output = F.gelu(lstm_output)
            lstm_output = self.dropout(lstm_output)

            # Apply the second linear layer to transform the output dimension of LSTM to 1
            embed2 = self.embeddings2[name](lstm_output)  # batch_size, seq_length, 1
            embeds.append(embed2)

        # Concatenate the embeddings of all features along the last dimension
        embeds = torch.cat(embeds, dim=-1)  # batch_size, seq_length, num_features
        return embeds


class StaticEncEmbedding(nn.Module):
    def __init__(self, numerical_features, embed_dim, max_len=1000, categorical_features=[], categorical_features_num=[], dropout=0.1):
        super().__init__()
        self.numerical_features = numerical_features
        self.embed_dim = embed_dim
        self.categorical_features = categorical_features
        self.num_features = len(numerical_features) + len(categorical_features)

        # numerical features
        self.numerical_embeddings1 = nn.ModuleDict({name: nn.Linear(1, 64, bias=True) for name in numerical_features})
        self.numerical_embeddings2 = nn.ModuleDict({name: nn.Linear(64, embed_dim, bias=True) for name in numerical_features})

        # Embedding layers for categorical features
        self.categorical_embeddings1 = nn.ModuleDict({name: nn.Embedding(categorical_features_num[idx_name], 64)
                                             for idx_name, name in enumerate(self.categorical_features)})
        self.categorical_embeddings2_linear = nn.ModuleDict({name: nn.Linear(64, embed_dim)
                                             for idx_name, name in enumerate(self.categorical_features)})

        # Masked values for masked language model
        self.masked_values = nn.Parameter(torch.randn(max_len, self.num_features), requires_grad=True)

        self.dropout = nn.Dropout(dropout)

    #     self.init_weights()
    #
    # def init_weights(self):
    #     # truncated normal distribution
    #     trunc_normal_(self.masked_values, std=.02)

    def forward(self, x, feature_order, masked_index=None):
        """
        Input:
            x: batch_size, num_features
            feature_order: list of feature names = numerical_features + categorical_names
            masked_index: batch_size, num_features
            masked_vector: num_features, embed_dim
        """

        num_bs, num_features = x.shape

        if masked_index is not None:
            masked_vector = self.masked_values[:num_bs, :]
            masked_index = masked_index.to(masked_vector.device)
            x = torch.where(masked_index, masked_vector, x)

        embeds = []
        for i, name in enumerate(feature_order):
            if name in self.numerical_features:
                embeds1 = self.numerical_embeddings1[name](x[..., i:i + 1])
                embeds1 = F.gelu(embeds1)
                embeds1 = self.dropout(embeds1)
                embeds2 = self.numerical_embeddings2[name](embeds1)
            elif name in self.categorical_features:
                embeds1 = self.categorical_embeddings1[name](x[..., i:i + 1].long()).squeeze(1)
                # embeds1 = F.gelu(embeds1)
                # embeds1 = self.dropout(embeds1)
                embeds2 = self.categorical_embeddings2_linear[name](embeds1)
            else:
                raise ValueError(f"Feature {name} not found in numerical_features or categorical_features")

            embeds.append(embeds2)
        embeds = torch.stack(embeds, dim=-1)
        embeds = torch.sum(embeds, dim=-1)  # --> batch_size, embed_dim
        # embeds = torch.nansum(embeds, dim=-1)  # --> batch_size, embed_dim

        return embeds

class StaticDecEmbedding(nn.Module):
    def __init__(self, numerical_features, embed_dim, categorical_features=[], categorical_features_num=[], dropout=0.1, add_input_noise=False):
        super().__init__()
        output_dim = 1
        if add_input_noise:
            output_dim = 2
        self.numerical_embeddings1 = nn.ModuleDict({name: nn.Linear(embed_dim, 64, bias=True) for name in numerical_features+categorical_features})
        self.numerical_embeddings2 = nn.ModuleDict({name: nn.Linear(64, output_dim, bias=True) for name in numerical_features})
        self.categorical_embeddings2 = nn.ModuleDict({name: nn.Linear(64, num_categories) for (name, num_categories) in zip(categorical_features, categorical_features_num)})
        self.categorical_features = categorical_features
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, feature_order, mode="train"):
        embeds = []
        static_variables_dec_index_start, static_variables_dec_index_end = [], []
        accumulator = 0
        for name in feature_order:
            embed1 = self.numerical_embeddings1[name](x)
            embed1 = F.gelu(embed1)
            embed1 = self.dropout(embed1)
            if name in self.categorical_features:
                embed2 = self.categorical_embeddings2[name](embed1).squeeze(1)
                if mode in ["test"]:
                    embed2 = F.softmax(embed2, dim=-1)
                    embed2 = torch.argmax(embed2, dim=1, keepdim=True)
            else:
                embed2 = self.numerical_embeddings2[name](embed1)
            embeds.append(embed2)
            static_variables_dec_index_start.append(accumulator)
            static_variables_dec_index_end.append(accumulator + embed2.shape[-1])
            # convert to list to torch tensor
            accumulator += embed2.shape[-1]

        # Concatenate the final outputs for each feature
        embeds = torch.concatenate(embeds, dim=-1)

        static_variables_dec_index_start = torch.tensor(static_variables_dec_index_start, dtype=torch.long).to(embeds.device)
        static_variables_dec_index_end = torch.tensor(static_variables_dec_index_end, dtype=torch.long).to(embeds.device)

        return embeds, static_variables_dec_index_start, static_variables_dec_index_end

class StaticDecEmbeddingLSTM(nn.Module):
    def __init__(self, numerical_features, embed_dim, categorical_features=[], categorical_features_num=[], dropout=0.4, add_input_noise=False):
        super().__init__()
        output_dim = 1
        if add_input_noise:
            output_dim = 2
        # Using LSTM for numerical features
        self.numerical_embeddings1 = nn.ModuleDict({name: nn.LSTM(embed_dim, 64, batch_first=True) for name in numerical_features})
        # Using linear layers for transforming LSTM output to 1 dimension for numerical features
        self.numerical_embeddings2 = nn.ModuleDict({name: nn.Linear(64, output_dim, bias=True) for name in numerical_features})
        # Linear layers for categorical features are unchanged
        self.categorical_embeddings1 = nn.ModuleDict({name: nn.Linear(embed_dim, 64, bias=True) for name in categorical_features})
        self.categorical_embeddings2 = nn.ModuleDict({name: nn.Linear(64, num_categories) for (name, num_categories) in zip(categorical_features, categorical_features_num)})
        self.categorical_features = categorical_features
        self.dropout = nn.Dropout(0.4)
        print("Fixed dropout rate of 0.4 for StaticDecEmbeddingLSTM")

    def forward(self, x, feature_order, mode="train"):
        embeds = []
        static_variables_dec_index_start, static_variables_dec_index_end = [], []
        accumulator = 0
        for name in feature_order:
            if name in self.categorical_features:
                # Process categorical features using linear layers
                embed1 = self.categorical_embeddings1[name](x)
                embed1 = F.gelu(embed1)
                embed1 = self.dropout(embed1)
                embed2 = self.categorical_embeddings2[name](embed1).squeeze(1)
                if mode == "test":
                    embed2 = F.softmax(embed2, dim=-1)
                    embed2 = torch.argmax(embed2, dim=1, keepdim=True)
            else:
                # Process numerical features using LSTM
                lstm_output, _ = self.numerical_embeddings1[name](x)  # batch_size, seq_length, 64
                lstm_output = F.gelu(lstm_output)
                lstm_output = self.dropout(lstm_output)
                embed2 = self.numerical_embeddings2[name](lstm_output)  # batch_size, seq_length, 1

            embeds.append(embed2)
            static_variables_dec_index_start.append(accumulator)
            static_variables_dec_index_end.append(accumulator + embed2.shape[-1])
            accumulator += embed2.shape[-1]

        # Concatenate the final outputs for each feature
        embeds = torch.cat(embeds, dim=-1)

        static_variables_dec_index_start = torch.tensor(static_variables_dec_index_start, dtype=torch.long).to(embeds.device)
        static_variables_dec_index_end = torch.tensor(static_variables_dec_index_end, dtype=torch.long).to(embeds.device)

        return embeds, static_variables_dec_index_start, static_variables_dec_index_end

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LinearEmbedding(nn.Module):
    def __init__(self, c_in, d_model, initrange=None):
        super(LinearEmbedding, self).__init__()
        self.initrange = initrange
        self.c_in = c_in
        self.d_model = d_model

        self.linear = nn.Linear(c_in, d_model, bias=False)

        # initial parameters.
        self.reset_parameters()

    def reset_parameters(self):
        if self.initrange is None:
            initrange = 1.0 / math.sqrt(self.d_model)
        else:
            initrange = self.initrange

        for weight in self.parameters():
            weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.linear(x)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super().__init__()

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.time_embedding = LinearEmbedding(c_in=3, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.value_embedding = LinearEmbedding(c_in=c_in, d_model=d_model)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.time_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)
