__all__ = ["IsochroneCMD"]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


def _clip_preserve_gradients(x, min_, max_):
    return x + lax.stop_gradient(jnp.clip(x, min_, max_) - x)


def _eval_poly(coeffs: ArrayLike, x: ArrayLike) -> jax.Array:
    """Horner evaluation of polynomial coefficients (highest power first,
    matching ``numpy.poly1d``'s convention)."""
    coeffs = jnp.asarray(coeffs)

    def body(acc, c):
        return acc * x + c, None

    result, _ = lax.scan(body, jnp.zeros_like(x, dtype=coeffs.dtype), coeffs)
    return result


class IsochroneCMD(dist.Distribution):
    """
    Joint (color, magnitude) density for a single-stellar-population (SSP)
    isochrone track, evaluated at a phi1-dependent apparent distance
    modulus -- following the stream CMD modeling approach of Starkman et al.
    2025 ("Stream Members Only"). Rather than a flexible density estimator
    (as used for the more data-rich background CMD; see ``FlowDensity`` /
    ``CalibratedFlowDensity``), the stream's photometry is modeled as a
    single old, metal-poor isochrone track (fixed shape -- generated once
    offline, see ``scripts/generate_stream_isochrone.py`` in ``gd1-dr3``)
    shifted along phi1 by a distance-modulus track, with intrinsic Gaussian
    scatter in color around the ridge line and a uniform marginal density in
    absolute magnitude over the isochrone's valid range.

    Follows the same conditioning convention as the rest of this codebase's
    distributions (``NormalSpline``, ``FlowDensity``, ...): ``x`` (here,
    phi1) is fixed at construction time but can be overridden by passing a
    new ``x`` to ``.log_prob()`` / ``.sample()``.

    Parameters
    ----------
    track_abs_mag
        Array of absolute-magnitude values (e.g. absolute Gaia G) along the
        isochrone, strictly monotonic (increasing OR decreasing), used as
        the spline "knots" for the ridge-line color(abs_mag) relation. See
        ``generate_stream_isochrone.restrict_to_monotonic_branch``.
    track_color
        Array of color values (e.g. BP-RP) at each ``track_abs_mag``.
    distmod_coeffs
        Coefficients (highest power first, as in ``numpy.poly1d``) of a
        polynomial giving the apparent distance modulus as a function of
        phi1 (degrees), e.g. the Valluri et al. 2024 GD-1 track:
        ``np.poly1d([1/64**2, 100/64**2, (50/64)**2 + 18.82 - 4.45])``.
    x
        Array of phi1 values (degrees) at which to evaluate the distance
        modulus.
    dm_offset (optional)
        Additive offset (mag) applied to the distance modulus, on top of
        ``distmod_coeffs(x)``. Defaults to 0 (i.e. trust the fixed
        Valluri+24 track exactly). Can be passed a numpyro prior (via
        ``coord_parameters``) to let the joint fit calibrate for any small,
        phi1-independent mismatch between the fixed literature distance
        track and this isochrone/photometric system.
    color_offset (optional)
        Additive offset (mag) applied to the isochrone's color, to absorb
        small systematics (e.g. reddening, metallicity, or model-isochrone
        mismatches) relative to the real data. Defaults to 0.
    color_scale
        Intrinsic Gaussian scatter (mag) in color around the (offset)
        isochrone ridge line, at fixed absolute magnitude. Captures
        photometric errors, unresolved binaries, and any real population
        width not captured by the single-age/single-metallicity track.
    spline_k (optional)
        Degree of the color(abs_mag) interpolating spline. Default 1
        (piecewise-linear), since isochrone tracks are typically sampled
        densely enough that higher-order interpolation isn't needed and a
        linear spline is guaranteed not to introduce spurious wiggles.
    """

    support = dist.constraints.real_vector

    def __init__(
        self,
        track_abs_mag: ArrayLike,
        track_color: ArrayLike,
        distmod_coeffs: ArrayLike,
        x: ArrayLike,
        dm_offset: ArrayLike = 0.0,
        color_offset: ArrayLike = 0.0,
        color_scale: ArrayLike = 0.05,
        spline_k: int = 1,
        validate_args: bool | None = None,
    ) -> None:
        x = jnp.asarray(x)
        super().__init__(
            batch_shape=x.shape, event_shape=(2,), validate_args=validate_args
        )

        self.track_abs_mag = jnp.asarray(track_abs_mag)
        self.track_color = jnp.asarray(track_color)
        self.distmod_coeffs = jnp.asarray(distmod_coeffs)
        self.x = x
        self.dm_offset = dm_offset
        self.color_offset = color_offset
        self.color_scale = color_scale
        self.spline_k = spline_k

        # InterpolatedUnivariateSpline requires strictly increasing knots;
        # isochrone tracks are typically ordered so abs_mag is *decreasing*
        # with increasing mass, so flip if needed. NOTE: this must be
        # implemented with `jnp.where` rather than a Python-level `if`, even
        # though `track_abs_mag`/`track_color` are conceptually "fixed"
        # (non-numpyro-sampled) data: `ModelComponent.make_dists` (and
        # therefore this constructor) gets called from inside numpyro/SVI
        # machinery such as `find_valid_initial_params`'s `lax.while_loop`,
        # which abstractly traces *everything* reachable inside it,
        # including closed-over "constant" arrays -- not just numpyro sample
        # sites. A Python `if` on a traced value raises
        # `TracerBoolConversionError` in that context (verified directly:
        # see this fix's commit message / test script for the reproduction).
        abs_mag = self.track_abs_mag
        color = self.track_color
        is_decreasing = abs_mag[-1] < abs_mag[0]
        self._abs_mag_sorted = jnp.where(is_decreasing, abs_mag[::-1], abs_mag)
        self._color_sorted = jnp.where(is_decreasing, color[::-1], color)

        self._color_spl = InterpolatedUnivariateSpline(
            self._abs_mag_sorted, self._color_sorted, k=self.spline_k
        )
        self._abs_mag_min = jnp.min(self.track_abs_mag)
        self._abs_mag_max = jnp.max(self.track_abs_mag)

    def _distmod(self, x: ArrayLike) -> jax.Array:
        return _eval_poly(self.distmod_coeffs, jnp.asarray(x)) + self.dm_offset

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array | Any:
        """
        Evaluates the log probability density for a batch of (color,
        magnitude) samples, e.g. ``(bp_rp, phot_g_mean_mag)``.

        Parameters
        ----------
        value
            Array of shape ``(..., 2)`` with columns ``(color, magnitude)``.
        x
            Array of phi1 values at which to evaluate the distance modulus.
            If not provided, the ``x`` values provided at initialization
            will be used.
        """
        x = self.x if x is None else jnp.asarray(x)
        value = jnp.asarray(value)
        color_obs, mag_obs = value[..., 0], value[..., 1]

        dm = self._distmod(x)
        abs_mag = mag_obs - dm

        clipped_abs_mag = _clip_preserve_gradients(
            abs_mag, self._abs_mag_min, self._abs_mag_max
        )
        pred_color = self._color_spl(clipped_abs_mag) + self.color_offset

        color_lp = dist.Normal(loc=pred_color, scale=self.color_scale).log_prob(
            color_obs
        )
        # Uniform marginal density in absolute magnitude over the track's
        # valid range -- see IsochroneCMD's docstring for why a full
        # population luminosity function isn't modeled here.
        mag_lp = -jnp.log(self._abs_mag_max - self._abs_mag_min)

        in_range = (abs_mag >= self._abs_mag_min) & (abs_mag <= self._abs_mag_max)
        return jnp.where(in_range, color_lp + mag_lp, -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        sample_shape: Any = (),
        x: ArrayLike | None = None,
    ) -> jax.Array | Any:
        """
        Draws (color, magnitude) samples: absolute magnitude uniform over
        the track's valid range, color drawn from a Gaussian around the
        (offset) isochrone ridge line at that absolute magnitude, then both
        shifted to apparent magnitude via the phi1-dependent distance
        modulus.
        """
        # NOTE: a per-call `x` must *fully* override the batch shape (matching
        # `log_prob`'s convention, and the sibling `NormalSpline.sample`),
        # not broadcast against `self.batch_shape` (the shape of whatever `x`
        # was passed at construction time). Mixing the two in with
        # `jnp.broadcast_shapes` was a bug: e.g. sampling one star at a time
        # via `jax.vmap(lambda k, x: dist.sample(k, x=x))(keys, phi1_array)`
        # -- exactly the pattern used to sanity-check model samples against
        # real data elsewhere in this pipeline -- would broadcast each
        # per-star scalar `x` back up against the *original*, full-length
        # construction-time `x`, silently producing a wrongly-shaped
        # `(len(original_x), 2)` output per vmap iteration instead of `(2,)`.
        x = self.x if x is None else jnp.asarray(x)
        shape = tuple(sample_shape) + x.shape

        key_mag, key_color = jax.random.split(key)
        abs_mag = jax.random.uniform(
            key_mag, shape, minval=self._abs_mag_min, maxval=self._abs_mag_max
        )
        pred_color = self._color_spl(abs_mag) + self.color_offset
        color = pred_color + self.color_scale * jax.random.normal(key_color, shape)

        dm = self._distmod(jnp.broadcast_to(x, shape))
        mag = abs_mag + dm

        return jnp.stack([color, mag], axis=-1)
