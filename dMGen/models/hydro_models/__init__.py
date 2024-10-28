import logging
import warnings
from abc import ABC
from typing import Dict, List

import torch

log = logging.getLogger(__name__)

# Disable prototype warnings and such
warnings.filterwarnings(action="ignore", category=UserWarning)



class RiverNetworkMatrix(torch.autograd.Function):
    """
    A custom autograd function for using sparse tensors with river routing
    """

    @staticmethod
    def forward(*args, **kwargs):
        """
        In the forward pass we receive a Tensor containing the
        input and return a Tensor containing the output.
        ctx is a context object that can be used
        to stash information for backward computation.
        You can cache arbitrary
        objects for
        """
        ctx, A_values, crow_indices, col_indices = args
        A_csr = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            A_values,
            dtype=torch.float64,
        )
        ctx.save_for_backward(
            A_values,
            A_csr,
            crow_indices,
            col_indices,
        )
        return A_csr

    @staticmethod
    def backward(*args):
        """
        In the backward pass, we receive a Tensor containing
        the gradient of the loss with respect to the output,
        and we need to compute the gradient of the loss
        with respect to the input.
        """

        def extract_csr_values(
            col_indices: torch.Tensor,
            crow_indices: torch.Tensor,
            dense_matrix: torch.Tensor,
        ) -> torch.Tensor:
            """
            Taking dense matrices and getting the sparse values
            :param col_indices:
            :param crow_indices:
            :param dense_matrix:
            :return:
            """
            crow_indices_list = crow_indices.tolist()
            col_indices_list = col_indices.tolist()
            rows = []
            cols = []
            for i in range(len(crow_indices_list) - 1):
                start, end = crow_indices_list[i], crow_indices_list[i + 1]
                these_cols = col_indices_list[start:end]
                these_rows = [i] * len(these_cols)
                rows.extend(these_rows)
                cols.extend(these_cols)
            values = dense_matrix[rows, cols]
            return values

        ctx, grad_output = args
        with torch.no_grad():
            A_values, A_csr, crow_indices, col_indices = ctx.saved_tensors
            grad_A_values = extract_csr_values(col_indices, crow_indices, grad_output)
        return grad_A_values, None, None, None


class PatternMapper:
    """
    map from data vector to non-zero elements of a sparse matrix
    data vector means the non-zero values organized in a convenient way,
    e.g., as diagonals which map into the matrix
    we make use of a generic full matrix filling operation (fullFillOp)
    to record that mapping relationship
    this mapping relationship is recorded as a mapping matrix to be reused
    sometimes you have legacy code that builds a big (but sparse)
    matrix and you don't want to figure out the insides
    this function then help to put data into the right place
    using your code fullFillOp we support adding some
    constant diagonals
    for example constant_diags=[1.0,2.0], constant_offsets=[0,1]
    """

    def __init__(
        self,
        fillOp,
        matrix_dim,
        constant_diags=None,
        constant_offsets=None,
        aux=None,
        indShift=0,
        device=None,
    ):
        # indShift=1 if the sparse matrix of a library start with index 1
        # starting from offset rather than 0,
        # this is to avoid 0 getting removed from to_sparse_csr
        offset = 1
        indVec = torch.arange(
            start=offset,
            end=matrix_dim + offset,
            dtype=torch.float64,
            device=device,
        )
        A = fillOp(indVec)  # it can return either full or sparse

        if not A.is_sparse_csr:
            A = A.to_sparse_csr()

        self.crow_indices = A.crow_indices()
        self.col_indices = A.col_indices()

        I_matrix = A.values().int() - offset
        # this mapping is: datvec(I(i)) --> sp.values(i)
        # this is equivalent to datvec * M
        # where M is a zero matrix with 1 at (I(i),i)
        C = torch.arange(0, len(I_matrix), device=device) + indShift
        indices = torch.stack((I_matrix, C), 0)
        ones = torch.ones(len(I_matrix), dtype=torch.float64, device=device)
        M_coo = torch.sparse_coo_tensor(
            indices, ones, dtype=torch.float64, device=device
        )
        self.M_csr = M_coo.to_sparse_csr()

    def map(self, datvec):
        return torch.matmul(datvec, self.M_csr).to_dense()

    def getSparseIndices(self):
        return self.crow_indices, self.col_indices

    @staticmethod
    def inverse_diag_fill(data_vector):
        """just an example test operation
        to fill in a matrix using datvec in a weird way"""
        # A = fillOp(vec) # A can be sparse in itself.
        n = data_vector.shape[0]
        # a slow matrix filling operation
        A = torch.zeros([n, n], dtype=data_vector.dtype)
        for i in range(n):
            A[i, i] = data_vector[n - 1 - i]
        return A

    @staticmethod
    def diag_aug(datvec, n, constant_diags, constant_offsets):
        datvec_aug = datvec.clone()
        # constant_offsets = [0]
        for j in range(len(constant_diags)):
            d = torch.zeros([n]) + constant_diags[j]
            datvec_aug = torch.cat((datvec_aug, d), nsdim(datvec))
        return datvec_aug


def nsdim(datvec):
    for i in range(datvec.ndim):
        if datvec[i] > 1:
            ns = i
            return ns
    return None


def denormalize(value: torch.Tensor, bounds: List[float]) -> torch.Tensor:
    output = (value * (bounds[1] - bounds[0])) + bounds[0]
    return output


class PhysicsModel(ABC, torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(PhysicsModel, self).__init__()

    def __setbounds__(self):
        raise NotImplementedError(
            "You need to implement a function to set parameter bounds"
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("The forward function must be implemented")
