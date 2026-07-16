__all__ = ["FlowDensity"]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from flowjax.distributions import AbstractDistribution as AbstractFlowJaxDistribution
from jax.typing import ArrayLike


class FlowDensity(dist.Distribution):
    """
    Wraps a (typically pretrained) ``flowjax`` normalizing flow so it can be used as
    a numpyro ``Distribution`` for one or more coordinates (e.g. a joint
    color-magnitude density).

    This follows the same conditioning convention as
    :class:`~stream_membership.distributions.normal_spline.NormalSpline`: the flow can
    be conditioned on some other coordinate (e.g. ``phi1``), and the conditioning
    array ``x`` is fixed at construction time but can be overridden by passing a new
    ``x`` when calling ``.log_prob()`` or ``.sample()``. This is needed, for example,
    when sampling a model component in ``sample_order`` where ``x`` (e.g. ``phi1``)
    is sampled first and its value must be threaded through to condition this flow.

    Unlike most of the other distributions in this module, the flow's internal
    parameters (i.e. the weights of the transformer/conditioner networks) are *not*
    exposed to numpyro through this wrapper, so they will not be updated during SVI or
    MCMC inference of the rest of the model. The flow is expected to be pretrained
    separately (e.g. by maximum likelihood on off-stream / background data) and then
    loaded here as a frozen density model -- mirroring how a pretrained background
    flow is used (with gradients disabled) in other normalizing-flow-based stream
    models.

    Parameters
    ----------
    flow
        A ``flowjax`` distribution representing the (optionally conditional) density
        of one or more coordinates, for example one built with
        ``flowjax.flows.masked_autoregressive_flow()``. This is treated as a fixed,
        frozen density: it is not registered as a numpyro sample site.
    x
        The conditioning variable(s) (e.g. an array of ``phi1`` values, one per data
        point), with shape ``(N,)`` or ``(N, *flow.cond_shape)``. If the wrapped flow
        is unconditional (``flow.cond_shape is None``), this should be left as
        ``None``.
    """

    def __init__(
        self,
        flow: AbstractFlowJaxDistribution,
        x: ArrayLike | None = None,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.flow = flow

        if flow.cond_shape is not None and x is None:
            msg = (
                "The wrapped flow is conditional (flow.cond_shape is "
                f"{flow.cond_shape}), so you must also pass a conditioning array `x`."
            )
            raise ValueError(msg)
        if flow.cond_shape is None and x is not None:
            msg = "The wrapped flow is unconditional, so `x` must be None."
            raise ValueError(msg)

        self.x = None if x is None else jnp.asarray(x)

        batch_shape = () if self.x is None else self._condition(self.x).shape[:-1]
        event_shape = tuple(flow.shape)

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def _condition(self, x: ArrayLike | None) -> jax.Array | None:
        """Reshape a conditioning array `x` to match `self.flow.cond_shape`."""
        if self.flow.cond_shape is None:
            return None

        x = self.x if x is None else jnp.asarray(x)
        if x is None:
            msg = "This flow is conditional: you must provide a conditioning array `x`."
            raise ValueError(msg)

        n_cond = len(self.flow.cond_shape)
        # If the trailing axes of x don't already match cond_shape, assume x has
        # shape (..., ) of "raw" per-sample conditioning values (e.g. phi1 with shape
        # (N,)) and add trailing axes so it broadcasts to (..., *cond_shape).
        if x.shape[-n_cond:] != tuple(self.flow.cond_shape):
            x = x.reshape(x.shape + (1,) * n_cond)
            x = jnp.broadcast_to(x, x.shape[:-n_cond] + tuple(self.flow.cond_shape))
        return x

    def sample(
        self,
        key: jax.Array,
        sample_shape: Any = (),
        x: ArrayLike | None = None,
    ) -> jax.Array | Any:
        """
        Draws samples from the (optionally conditional) flow.

        Parameters
        ----------
        key
            JAX random number generator key.
        sample_shape
            Shape of the sample.
        x
            Conditioning array. If not provided, the ``x`` values provided at
            initialization will be used.
        """
        condition = self._condition(x)
        return self.flow.sample(key, sample_shape=sample_shape, condition=condition)

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array | Any:
        """
        Evaluates the log probability density for a batch of samples given by value.

        Parameters
        ----------
        value
            Array of samples to evaluate the log probability for.
        x
            Conditioning array. If not provided, the ``x`` values provided at
            initialization will be used.
        """
        condition = self._condition(x)
        return self.flow.log_prob(jnp.asarray(value), condition)

    @property
    def support(self):
        if len(self.event_shape) == 0:
            return dist.constraints.real
        return dist.constraints.real_vector
