"""Base class and contract for physics models added to 𝛿MG.

Drop a model in this directory and 𝛿MG will find it: ``import_phy_model``
maps the ``model.phy.name`` config entry to a file here by converting the
class name from CamelCase to snake_case (``MyModel`` -> ``my_model.py``).
Models shipped by `HydroDL2 <https://github.com/mhpi/hydrodl2>`_ are resolved
first; anything not found there is loaded from this package.

Subclassing :class:`BasePhysicsModel` is optional -- the loader only requires
a ``torch.nn.Module`` -- but it documents the interface the rest of 𝛿MG
relies on and supplies the boilerplate every model would otherwise repeat.
"""

from typing import Any, Optional

import torch


class BasePhysicsModel(torch.nn.Module):
    """Interface for a differentiable physics model in 𝛿MG.

    A physics model receives forcings plus a tensor of parameters learned by
    a neural network, and returns a dict of predicted fluxes. It owns its own
    warm-up handling; see the forward contract below.

    Subclasses keep their own ``__init__`` and only need to call
    ``super().__init__()``.

    The forward contract
    --------------------
    ``forward(x_dict, parameters)`` returns ``dict[str, torch.Tensor]`` whose
    time-major entries cover exactly ``nsteps - self.warmup`` timesteps, where
    ``nsteps`` is the length of the forcing window it was handed.

    This matters: 𝛿MG trims the *target* by ``model.warmup`` and then expects
    predictions to line up with no further adjustment. A model that returns
    the full window instead will silently score warm-up days against real
    observations. Two strategies satisfy the contract:

    - simulate the warm-up window separately (ideally under
      ``torch.no_grad()``) to spin up storages, then run the scored window --
      the warm-up steps never enter the outputs; or
    - simulate the whole window and drop the leading ``self.warmup`` steps
      with :meth:`trim_warmup` before returning.

    One key of the returned dict must match ``config['train']['target']`` so
    the trainer can find the prediction to score.

    Required attributes
    -------------------
    learnable_param_count
        How many values per timestep the neural network must emit for this
        model. The NN's output layer is sized from this, so changing it
        invalidates existing checkpoints.
    parameter_bounds
        ``{name: [lower, upper]}`` physical ranges, used to map the network's
        normalized outputs onto physical values.
    warmup
        Number of leading timesteps used to spin up internal storages.

    Example
    -------
    ::

        class MyModel(BasePhysicsModel):
            def __init__(self, config=None, device=None):
                super().__init__()
                self.device = device or 'cpu'
                self.nmul = config.get('nmul', 1)
                self.warmup = config.get('warmup', 0)
                self.state_names = ('storage',)
                self.parameter_bounds = {'k': [0.0, 1.0]}
                self.learnable_param_count = len(self.parameter_bounds) * self.nmul

            def forward(self, x_dict, parameters):
                nsteps = x_dict['x_phy'].shape[0]
                fluxes = self._simulate(x_dict, parameters)
                return self.trim_warmup(fluxes, self.warmup, nsteps)
    """

    #: Value every internal storage is initialized to. Zero is avoided so that
    #: divisions and powers taken on a fresh storage stay finite.
    initial_state_value: float = 0.001

    def __init__(self) -> None:
        super().__init__()
        self.states: Optional[tuple[torch.Tensor, ...]] = None
        self._state_cache: Optional[tuple[torch.Tensor, ...]] = None

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the model.

        Parameters
        ----------
        x_dict
            Input data. ``x_dict['x_phy']`` holds forcings shaped
            ``[nsteps, ngrid, n_forcings]``; models may read extra keys
            (basin area, elevation, ...) that their data loader provides.
        parameters
            Normalized parameters from the neural network, to be mapped onto
            :attr:`parameter_bounds`.

        Returns
        -------
        dict[str, torch.Tensor]
            Predicted fluxes. Time-major entries cover ``nsteps - warmup``
            steps; see the class docstring.
        """
        raise NotImplementedError

    @staticmethod
    def trim_warmup(
        outputs: dict[str, torch.Tensor],
        pred_cutoff: int,
        nsteps: int,
    ) -> dict[str, torch.Tensor]:
        """Drop ``pred_cutoff`` leading timesteps from time-major outputs.

        Only tensors whose leading dimension is ``nsteps`` are trimmed, so
        outputs that have already collapsed the time axis (a baseflow index
        summed over time, say) pass through untouched and callers do not have
        to maintain a list of exceptions.

        Parameters
        ----------
        outputs
            Model outputs.
        pred_cutoff
            Number of leading timesteps to drop. Values <= 0 are a no-op.
        nsteps
            Length of the simulated window, used to identify time-major
            tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Outputs with the warm-up period removed.
        """
        if pred_cutoff <= 0:
            return outputs

        trimmed = {}
        for key, value in outputs.items():
            is_time_major = (
                torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == nsteps
            )
            trimmed[key] = value[pred_cutoff:] if is_time_major else value
        return trimmed

    def _init_states(self, ngrid: int) -> tuple[torch.Tensor, ...]:
        """Initialize every internal storage to :attr:`initial_state_value`.

        Requires ``state_names``, ``nmul``, and ``device`` to be set.

        Parameters
        ----------
        ngrid
            Number of basins/catchments in the batch.

        Returns
        -------
        tuple[torch.Tensor, ...]
            One ``[ngrid, nmul]`` tensor per entry in ``state_names``.
        """

        def make_state():
            return torch.full(
                (ngrid, self.nmul),
                self.initial_state_value,
                dtype=torch.float32,
                device=self.device,
            )

        return tuple(make_state() for _ in range(len(self.state_names)))

    def get_states(self) -> Optional[tuple[torch.Tensor, ...]]:
        """Return the internal states cached by the last forward pass.

        Used for sequential simulation, where a run resumes from where the
        previous one stopped.

        Returns
        -------
        tuple[torch.Tensor, ...] or None
            One tensor per entry in ``state_names``, or ``None`` if the model
            has not been run yet.
        """
        return self._state_cache

    def load_states(self, states: tuple[torch.Tensor, ...]) -> None:
        """Load internal states, moved to the model's device and dtype.

        Parameters
        ----------
        states
            One tensor per entry in ``state_names``.
        """
        for state in states:
            if not isinstance(state, torch.Tensor):
                raise ValueError("Each element in `states` must be a tensor.")
        nstates = len(self.state_names)
        if not (isinstance(states, tuple) and len(states) == nstates):
            raise ValueError(f"`states` must be a tuple of {nstates} tensors.")

        self.states = tuple(
            s.detach().to(self.device, dtype=torch.float32) for s in states
        )

    def _descale_phy_dy_parameters(
        self,
        phy_dy_params: torch.Tensor,
        dy_list: list[str],
    ) -> dict[str, torch.Tensor]:
        """Descale the time-varying physical parameters.

        Dynamic and static parameters are descaled separately on purpose:
        only the dynamic ones carry a time axis, so the network head emitting
        them can be much narrower. Folding the static parameters back into
        this tensor would make the network emit a full time series for values
        that never change, which is the main memory cost in distributed runs.

        Parameters
        ----------
        phy_dy_params
            Normalized dynamic parameters, ``[nsteps, ngrid, n_dynamic, nmul]``.
        dy_list
            Names of the dynamic parameters, in channel order.
        """
        raise NotImplementedError

    def _descale_phy_stat_parameters(
        self,
        phy_stat_params: torch.Tensor,
        stat_list: list[str],
    ) -> dict[str, torch.Tensor]:
        """Descale the time-invariant physical parameters.

        Parameters
        ----------
        phy_stat_params
            Normalized static parameters, ``[ngrid, n_static, nmul]`` -- note
            the absence of a time axis.
        stat_list
            Names of the static parameters, in channel order.
        """
        raise NotImplementedError
