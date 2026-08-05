__all__ = ["LinearGradient1D", "LinearGradientSpline"]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


def _clip_preserve_gradients(x, min_, max_):
    return x + lax.stop_gradient(jnp.clip(x, min_, max_) - x)


class LinearGradient1D(dist.Distribution):
    """
    A distribution on the interval [low, high] with a linear density gradient:

        pdf(x) = (1 + a * u(x)) / (high - low)

    where ``u(x) = 2 * (x - low) / (high - low) - 1`` is ``x`` rescaled to
    ``[-1, 1]``, and ``a`` (in ``(-1, 1)``) sets the fractional density contrast
    between the two ends of the interval: the density at ``x=high`` is
    ``(1 + a)`` times the mean density (``1 / (high - low)``), and the density at
    ``x=low`` is ``(1 - a)`` times the mean density. ``a=0`` recovers the Uniform
    distribution on ``[low, high]`` exactly, so, unlike a heavily truncated
    Normal, this distribution does not require an extreme, poorly-conditioned
    parameter regime to represent a shallow or near-uniform density gradient.
    """

    arg_constraints = {"a": dist.constraints.interval(-1, 1)}

    def __init__(
        self,
        a: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        validate_args=None,
    ) -> None:
        """
        Parameters
        ----------
        a
            Tilt parameter that must be in ``(-1, 1)``. ``a=0`` is equivalent to a
            Uniform distribution on ``[low, high]``.
        low
            Lower bound of the distribution.
        high
            Upper bound of the distribution.
        """
        self.a = jnp.asarray(a)
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)

        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.a), jnp.shape(self.low), jnp.shape(self.high)
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def support(self):
        return dist.constraints.interval(self.low, self.high)

    def _u(self, value: ArrayLike) -> jax.Array:
        return 2 * (value - self.low) / (self.high - self.low) - 1

    def log_prob(self, value: ArrayLike) -> jax.Array:
        value = jnp.asarray(value)
        u = self._u(value)
        log_prob = jnp.log1p(self.a * u) - jnp.log(self.high - self.low)
        return jnp.where(self.support.check(value), log_prob, -jnp.inf)

    def icdf(self, p: ArrayLike) -> jax.Array:
        """
        Inverts the CDF (in the rescaled coordinate ``s = (x - low) / (high -
        low) in [0, 1]``), ``F(s) = a * s**2 + (1 - a) * s``, via the quadratic
        formula. The ``a -> 0`` limit (where the quadratic formula would divide
        by zero) is special-cased to the linear solution ``s = p``.
        """
        a = self.a
        p = jnp.asarray(p)

        is_linear = jnp.abs(a) < 1e-8
        # Avoid division by (near-)zero in the unselected branch -- `jnp.where`
        # evaluates both branches, and a literal 0 in the denominator would
        # otherwise poison gradients even though this branch is discarded.
        safe_denom = jnp.where(is_linear, 1.0, 2 * a)

        disc = (1 - a) ** 2 + 4 * a * p
        disc = jnp.clip(disc, 0.0, None)  # guard tiny negative values from roundoff
        s_quad = (-(1 - a) + jnp.sqrt(disc)) / safe_denom
        s_lin = p

        s = jnp.where(is_linear, s_lin, s_quad)
        return self.low + s * (self.high - self.low)

    def sample(self, key: jax.Array, sample_shape: Any = ()) -> jax.Array:
        shape = tuple(sample_shape) + self.batch_shape
        p = jax.random.uniform(key, shape=shape)
        return self.icdf(p)


class LinearGradientSpline(dist.Distribution):
    def __init__(
        self,
        a_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        spline_k: int = 3,
        clip_a: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """
        Represents a `LinearGradient1D` distribution where the tilt parameter
        `a` is controlled by a spline that is evaluated at some other parameter
        values x. In other words, this distribution is conditional on x.

        Parameters
        ----------
        a_vals
            Array of tilt parameter (`a`) values at the knot locations.
        knots
            Array of spline knot locations.
        x
            Array of x values at which to evaluate the spline.
        low
            Lower bound of the distribution.
        high
            Upper bound of the distribution.
        spline_k (optional)
            Degree of the spline.
        clip_a (optional)
            If specified, clips the spline-interpolated tilt parameter `a` into
            this range (using a straight-through-gradient trick, so forward
            values are clipped but gradients still flow as if unclipped). Since
            `a` must stay in `(-1, 1)` for the density to be valid, and nothing
            otherwise stops an optimizer (e.g. during SVI) from pushing the
            spline-interpolated `a` outside that range, it is recommended to set
            this to something like `(-0.999, 0.999)` in practice.
        """
        x = jnp.asarray(x)
        super().__init__(batch_shape=x.shape, event_shape=())

        self.knots = jnp.array(knots)
        self.low = low
        self.high = high
        self.clip_a = tuple(clip_a)

        self.spline_k = int(spline_k)
        self.x = x
        self.a_vals = jnp.array(a_vals)

        if self.a_vals.ndim == 0:
            self._a_spl = lambda _: self.a_vals
        else:
            self._a_spl = InterpolatedUnivariateSpline(
                self.knots,
                self.a_vals,
                k=self.spline_k,
                endpoints="not-a-knot",
            )

    def _make_helper_dist(self, x: ArrayLike | None = None) -> LinearGradient1D:
        x = self.x if x is None else x
        a = _clip_preserve_gradients(self._a_spl(x), *self.clip_a)
        return LinearGradient1D(a=a, low=self.low, high=self.high)

    def sample(
        self,
        key: jax.Array,
        sample_shape: Any = (),
        x: ArrayLike | None = None,
    ) -> jax.Array | Any:
        """
        Draws samples from the distribution.

        Parameters
        ----------
        key
            JAX random number generator key.
        sample_shape
            Shape of the sample.
        x
            Array of x values at which to evaluate the spline. If not provided,
            the x values provided at initialization will be used.
        """
        helper = self._make_helper_dist(x)
        return helper.sample(key=key, sample_shape=sample_shape)

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array | Any:
        """
        Evaluates the log probability density for a batch of samples given by
        value.

        Parameters
        ----------
        value
            Array of samples to evaluate the log probability for.
        x
            Array of x values at which to evaluate the spline. If not provided,
            the x values provided at initialization will be used.
        """
        helper = self._make_helper_dist(x)
        return helper.log_prob(value)

    @property
    def support(self):
        return dist.constraints.interval(self.low, self.high)
