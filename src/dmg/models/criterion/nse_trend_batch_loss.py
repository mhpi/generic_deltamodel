from typing import Any, Optional, Union

import numpy as np
import torch

from dmg.models.criterion.base import BaseCriterion


class NseTrendBatchLoss(BaseCriterion):
    """NSE loss weighted by |dQ/dt| — emphasizes hydrograph transitions.

    At each timestep the squared residual is multiplied by
    ``|Q_obs(t) - Q_obs(t-1)| + base_weight`` before basin-std
    normalization.  This up-weights rising/falling limb errors
    (transitions) relative to steady-state periods.

    ``base_weight`` (default 0.1) ensures that even flat periods
    receive *some* gradient so the expert does not completely
    ignore them.

    Intended use: train a "transition specialist" expert in a Multiple
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
        - base_weight: Minimum weight for flat periods. Default 0.1.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: Union[torch.Tensor, float],
    ) -> None:
        super().__init__(config, device)
        self.name = 'Batch Trend-Weighted NSE Loss'
        self.config = config
        self.device = device

        self.base_weight = kwargs.get(
            'base_weight',
            config.get('base_weight', 0.1),
        )

        try:
            y_obs = kwargs['y_obs']
            if isinstance(y_obs, torch.Tensor):
                y_obs = y_obs[:, :, 0].detach().to(torch.float64).cpu().numpy()
            else:
                y_obs = y_obs[:, :, 0]
            self.std = np.nanstd(y_obs, axis=0)
        except KeyError as e:
            raise KeyError("'y_obs' is not provided in kwargs") from e

        self.eps = kwargs.get('eps', config.get('eps', 0.1))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute trend-weighted NSE loss.

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
            n_timesteps = target.shape[0]
            std_batch = torch.tensor(
                np.tile(self.std[sample_ids].T, (n_timesteps, 1)),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )

            # Compute |dQ/dt| from observations (detached — no gradient).
            # First timestep gets the same weight as the second.
            with torch.no_grad():
                tgt_filled = target.clone()
                tgt_filled[torch.isnan(tgt_filled)] = 0.0
                dq = torch.abs(tgt_filled[1:] - tgt_filled[:-1])  # [T-1, N]
                dq = torch.cat([dq[:1], dq], dim=0)  # [T, N]
                # Normalize dq to [0, 1] per basin to keep loss scale stable.
                dq_max = dq.max(dim=0, keepdim=True).values.clamp(min=1e-6)
                dq_norm = dq / dq_max  # [T, N]
                trend_weight = dq_norm + self.base_weight  # [T, N]

            mask = ~torch.isnan(target)
            p_sub = prediction[mask]
            t_sub = target[mask]
            std_sub = std_batch[mask]
            w_sub = trend_weight[mask]

            sq_res = (p_sub - t_sub) ** 2
            norm_res = sq_res / (std_sub + self.eps) ** 2
            weighted_res = norm_res * w_sub

            loss = torch.mean(weighted_res)
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss
