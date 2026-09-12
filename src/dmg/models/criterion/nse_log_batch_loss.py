from typing import Any, Optional, Union

import numpy as np
import torch

from dmg.models.criterion.base import BaseCriterion


class NseLogBatchLoss(BaseCriterion):
    """NSE loss on log-transformed flows — emphasizes low-flow accuracy.

    Applies ``log(Q + eps_log)`` to both predictions and observations
    before computing the standard batch-NSE loss.  Because the log
    transform compresses high flows and expands low flows, this loss
    penalizes baseflow errors much more heavily than :class:`NseBatchLoss`.

    Intended use: train a "baseflow specialist" expert in a Multiple
    Choice Learning (MCL) ensemble.

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        - y_obs: Tensor/array of observations to compute stats. (Required)
        - eps: Stability term for std normalization. Default 0.1.
        - eps_log: Offset added before log transform. Default 0.01.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: Union[torch.Tensor, float],
    ) -> None:
        super().__init__(config, device)
        self.name = 'Batch Log-NSE Loss'
        self.config = config
        self.device = device

        self.eps_log = kwargs.get('eps_log', config.get('eps_log', 0.01))

        try:
            y_obs = kwargs['y_obs']
            if isinstance(y_obs, torch.Tensor):
                y_obs = y_obs[:, :, 0].detach().to(torch.float64).cpu().numpy()
            else:
                y_obs = y_obs[:, :, 0]
            # Compute std in log-space for normalization.
            y_log = np.log(np.clip(y_obs, a_min=self.eps_log, a_max=None))
            self.std = np.nanstd(y_log, axis=0)
        except KeyError as e:
            raise KeyError("'y_obs' is not provided in kwargs") from e

        self.eps = kwargs.get('eps', config.get('eps', 0.1))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-NSE loss.

        Parameters
        ----------
        y_pred
            Predicted values.
        y_obs
            Observed values.
        **kwargs
            - sample_ids: basin indices in this batch. (Required)

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        prediction, target = self._format(y_pred, y_obs)

        try:
            sample_ids = kwargs['sample_ids'].astype(int)
        except KeyError as e:
            raise KeyError("'sample_ids' is not provided in kwargs") from e

        if len(target) > 0:
            # Log-transform both predictions and targets.
            pred_log = torch.log(prediction.clamp(min=self.eps_log))
            tgt_log = torch.log(target.clamp(min=self.eps_log))

            n_timesteps = target.shape[0]
            std_batch = torch.tensor(
                np.tile(self.std[sample_ids].T, (n_timesteps, 1)),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )

            mask = ~torch.isnan(target)
            p_sub = pred_log[mask]
            t_sub = tgt_log[mask]
            std_sub = std_batch[mask]

            sq_res = (p_sub - t_sub) ** 2
            norm_res = sq_res / (std_sub + self.eps) ** 2

            loss = torch.mean(norm_res)
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss
