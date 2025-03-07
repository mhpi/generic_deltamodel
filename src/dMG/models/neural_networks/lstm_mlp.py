import torch

from .ann import AnnModel
from .cudnn_lstm import CudnnLstmModel


class LstmMlpModel(torch.nn.Module):
    """LSTM-MLP model for multi-scale learning."""
    def __init__(self, *, nx1, ny1,hiddeninv1,nx2, ny2,hiddeninv2,dr1=0.5,dr2=0.5):
        super().__init__()
        self.lstminv = CudnnLstmModel(
            nx=nx1, ny=ny1, hiddenSize=hiddeninv1, dr=dr1)
        
        self.Ann = AnnModel(
            nx=nx2, ny=ny2, hiddenSize=hiddeninv2, dropout_rate=dr2)

    def forward(self, z1,z2):
        Lstm_para = self.lstminv(z1) # dim: Time, Gage, Para

        Ann_para = self.Ann(z2)


        return [ torch.sigmoid(Lstm_para),Ann_para]
