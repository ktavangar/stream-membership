import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from numpyro.distributions import constraints
from numpyro.distributions.constraints import _IndependentConstraint, _Interval
from numpyro.distributions.util import promote_shapes

__all__ = ["TruncatedGridGMM"]


class _LeftExtendedReal(constraints.Constraint):
    """
    Any number in the interval [-inf, inf).
    """

    def __call__(self, x):
        return (x == x) & (x != float("inf"))

    def feasible_like(self, prototype):
        return jnp.zeros_like(prototype)


class _RightExtendedReal(constraints.Constraint):
    """
    Any number in the interval (-inf, inf].
    """

    def __call__(self, x):
        return (x == x) & (x != float("-inf"))

    def feasible_like(self, prototype):
        return jnp.zeros_like(prototype)


left_extended_real = _LeftExtendedReal()
right_extended_real = _RightExtendedReal()


class _IntervalVector(_IndependentConstraint):
    def __init__(self, lower_bound, upper_bound):
        super().__init__(_Interval(lower_bound, upper_bound), 1)


interval_vector = _IntervalVector


class TruncatedGridGMM(dist.mixtures.MixtureSameFamily):
    """
    A Gaussian Mixture Model where the components are fixed to their input locations and
    there are no covariances (but each dimension can have different scales / standard
    deviations). The model can also be truncated to a rectangular region.
    """

    arg_constraints = {
        "locs": constraints.real,
        "scales": constraints.positive,
        "low": left_extended_real,
        "high": right_extended_real,
    }
    reparametrized_params = ["locs", "scales", "low", "high"]

    def __init__(
        self,
        mixing_distribution,
        locs=0.0,
        scales=1.0,
        low=None,
        high=None,
        *,
        validate_args=True,
    ):
        dist.mixtures._check_mixing_distribution(mixing_distribution)

        if low is None:
            low = -float("inf")
        if high is None:
            high = float("inf")

        batch_shape = lax.broadcast_shapes(jnp.shape(locs), jnp.shape(scales))
        if validate_args:
            if len(batch_shape) != 2:
                raise ValueError("TODO: Bork")

        self.locs, self.scales = promote_shapes(
            jnp.array(locs),
            jnp.array(scales),
            shape=batch_shape,
        )
        self._K, self._D = batch_shape

        locs = jnp.broadcast_to(self.locs, batch_shape)
        scales = jnp.broadcast_to(self.scales, batch_shape)

        self.low = jnp.array(low)
        self.high = jnp.array(high)
        low = jnp.broadcast_to(self.low, batch_shape[1:])
        high = jnp.broadcast_to(self.high, batch_shape[1:])

        covs = jnp.zeros(batch_shape + (batch_shape[1],))
        idx = jnp.arange(batch_shape[1])
        covs = covs.at[:, idx, idx].set(scales**2)
        component_distribution = dist.MultivariateNormal(
            locs, covariance_matrix=covs, validate_args=validate_args
        )

        super().__init__(
            mixing_distribution, component_distribution, validate_args=validate_args
        )

        self._support = interval_vector(low, high)

        # Pre-compute the amount of probability mass that is inside the censored region
        self._log_diff_tail_probs = jnp.zeros(batch_shape[0])
        for k in range(batch_shape[0]):  # K Gaussian components
            for d in range(batch_shape[1]):  # D dimensions
                norm = dist.Normal(locs[k, d], scales[k, d])
                sign = jnp.where(locs[k, d] >= low[d], 1.0, -1.0)

                _tail_prob_at_low = norm.cdf(locs[k, d] - sign * (locs[k, d] - low[d]))
                _tail_prob_at_high = norm.cdf(
                    locs[k, d] - sign * (locs[k, d] - high[d])
                )

                self._log_diff_tail_probs = self._log_diff_tail_probs.at[k].add(
                    jnp.log(sign * (_tail_prob_at_high - _tail_prob_at_low))
                )

    @constraints.dependent_property
    def support(self):
        return self._support

    def component_log_probs(self, value):
        value = jnp.expand_dims(value, self.mixture_dim)
        component_log_probs = (
            self.component_distribution.log_prob(value) - self._log_diff_tail_probs
        )
        return jax.nn.log_softmax(self.mixing_distribution.logits) + component_log_probs

    def sample(self, *args, **kwargs):
        raise NotImplementedError("sample() is not implemented yet")