import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

__all__ = ["IndependentGMM"]


class IndependentGMM(dist.MixtureSameFamily):
    def __init__(
        self,
        mixing_distribution: dist.CategoricalLogits | dist.CategoricalProbs,
        locs: ArrayLike = 0.0,
        scales: ArrayLike = 1.0,
        low: ArrayLike | None = None,
        high: ArrayLike | None = None,
        *,
        validate_args=True,
    ):
        """
        A Gaussian Mixture Model where the components are fixed to their input locations
        and there are no covariances (but each dimension can have different scales /
        standard deviations).

        Parameters
        ----------
        mixing_distribution
            Distribution over the mixture components.
        locs
            Array of means for each component. This should have shape (D, K) where D is
            the dimensionality of the data and K is the number of mixture components.
        scales
            Array of standard deviations for each component. This should have shape (D,
            K) where D is the dimensionality of the data and K is the number of mixture
            components.
        low
            Lower bounds for each dimension. This should either be a scalar or have
            shape (D,) where D is the dimensionality of the data.
        high
            Upper bounds for each dimension. This should either be a scalar or have
            shape (D,) where D is the dimensionality of the data.
        """
        # K = mixture components, D = dimensions
        # - event_shape is the dimensionality of the data - number of dependent
        #   coordinates, i.e., "D" in the below
        # - batch_shape is the number of independent dimensions - here "K"
        combined_shape = jax.lax.broadcast_shapes(jnp.shape(locs), jnp.shape(scales))
        if len(combined_shape) != 2:
            msg = (
                f"locs and scales must have 2 axes, but got {len(combined_shape)}. The "
                "shape must be: (D, K) where D is the dimensionality of the data and K "
                "is the number of mixture components."
            )
            raise ValueError(msg)
        self._D, self._K = combined_shape

        component_kwargs = {"loc": locs, "scale": scales}
        if low is not None:
            component_kwargs["low"] = low
        if high is not None:
            component_kwargs["high"] = high

        component = dist.TruncatedNormal(**component_kwargs)
        component._batch_shape = (self._D, self._K)
        self._low = low
        self._high = high

        # NOTE: we deliberately do *not* call `super().__init__()`
        # (`MixtureSameFamily.__init__`) here. As of numpyro>=0.20,
        # `MixtureSameFamily.__init__` asserts that `component_distribution.support`
        # is a `ParameterFreeConstraint`, which a bounded `TruncatedNormal` never
        # satisfies (its support is a parameterized `Interval(low, high)`). That
        # check exists so the base class's generic `log_prob`/`cdf`/etc. can assume
        # a fixed support across components, but `IndependentGMM` overrides all of
        # those (see `log_prob`, `component_log_probs`, `support` below) and
        # explicitly masks out-of-bounds values itself, so the restriction doesn't
        # apply to us. Instead, we replicate the (small) subset of
        # `MixtureSameFamily.__init__`'s bookkeeping that we actually rely on,
        # skipping straight to `Distribution.__init__`. This mirrors the numpyro
        # 0.19 and 0.20 implementations identically apart from the added assert.
        n_components = getattr(mixing_distribution, "probs", None)
        n_components = (
            n_components.shape[-1]
            if n_components is not None
            else mixing_distribution.logits.shape[-1]
        )
        if component.batch_shape[-1] != n_components:
            msg = (
                "Component distribution batch shape last dimension "
                f"(size={component.batch_shape[-1]}) needs to correspond to the "
                f"mixture_size={n_components}!"
            )
            raise ValueError(msg)

        dist.Distribution.__init__(
            self, batch_shape=(), event_shape=(self._D,), validate_args=validate_args
        )
        self._mixing_distribution = mixing_distribution
        self._component_distribution = component
        self._mixture_size = n_components
        self._dim_dim = -2

    @property
    def mixture_dim(self):
        return -1

    @property
    def support(self):
        # TODO: it's possible this is not correct. The component distribution support
        # may not be a vector interval like it needs to be? Anyways, if we see issues
        # with using this distribution, audit the support!
        if self.component_distribution.support is not None:
            return self.component_distribution.support
        return dist.constraints.real

    def component_log_probs(self, value: ArrayLike) -> jax.Array:
        value = jnp.array(value)
        value = jnp.atleast_2d(value.T).T

        if value.shape[-1] != self._D:
            msg = (
                "The input array must have the same number of coordinate dimensions "
                f"as the distribution. Expected {self._D}, got {value.shape}."
            )
            raise ValueError(msg)

        tmp = jnp.expand_dims(value, self.mixture_dim)

        # Two distinct NaN-gradient hazards are handled here, both stemming from
        # the same root cause: real data can easily contain a point outside the
        # (still-converging, during early SVI steps) truncation bounds, and
        # naively computing log_prob there and masking the *output* with
        # `jnp.where` is not gradient-safe.
        #
        # (1) Evaluating TruncatedNormal.log_prob() directly at an out-of-bounds
        #     point gives the mathematically correct forward value (-inf), but
        #     its *gradient* w.r.t. loc/scale is NaN, and `jnp.where` does not
        #     protect against NaN gradients flowing through the branch it
        #     doesn't select. Fix: clip the input into the valid support (a
        #     "safe" placeholder that can never produce a NaN gradient) before
        #     calling log_prob, so it's never evaluated at an invalid point.
        # (2) `low`/`high` are shared across all K mixture components, so a
        #     point outside bounds in even one dimension is out-of-support for
        #     *every* component simultaneously. If we mask with a literal
        #     `-jnp.inf`, `log_prob`'s `logsumexp` below is then taken over a
        #     vector that is entirely -inf, which is a genuine 0/0 in
        #     logsumexp's softmax-gradient formula (NaN), independent of fix
        #     (1) above. Fix: mask with a large-but-finite sentinel instead of
        #     literal -inf, so it's numerically indistinguishable from zero
        #     probability but keeps logsumexp's gradient well-defined.
        # See: https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where
        low = jnp.asarray(-jnp.inf) if self._low is None else jnp.asarray(self._low)
        high = jnp.asarray(jnp.inf) if self._high is None else jnp.asarray(self._high)
        # `tmp` has shape (..., D, K) (K = number of mixture components,
        # broadcast via `mixture_dim=-1`). `low`/`high` describe a per-
        # dimension (D) bound that doesn't depend on K, so they need a
        # trailing size-1 axis to broadcast against `tmp`'s last axis. But
        # `self._low`/`self._high` may *already* carry that trailing axis:
        # `__init__` requires shape (D, 1) (not bare (D,)) for `low`/`high`
        # to broadcast correctly against `loc`/`scale`'s (D, K) shape when
        # constructing the underlying `TruncatedNormal` (a raw (D,) array
        # would wrongly align against the *K* axis there instead of *D*, and
        # fail unless D happened to equal K). So only append a new axis here
        # if `low`/`high` are still in bare 1-D (D,) form (e.g. if a caller
        # passes a plain list per the docstring) -- appending one
        # unconditionally double-adds an axis for the (D, 1) case already
        # required by construction, producing an unbroadcastable (D, 1, 1)
        # array (only surfaces once this GMM is evaluated with D > 1, e.g.
        # inside a `ComponentMixtureModel`, since a D=1 mismatch is masked by
        # broadcasting's leading-1 rule).
        low = low.reshape(low.shape + (1,)) if low.ndim == 1 else low
        high = high.reshape(high.shape + (1,)) if high.ndim == 1 else high
        safe_tmp = jnp.clip(tmp, low, high)
        component_log_probs = self.component_distribution.log_prob(safe_tmp)

        value = jnp.expand_dims(value, axis=-1)
        neg_inf_sentinel = jnp.asarray(-1e10, dtype=component_log_probs.dtype)
        return jnp.where(
            self.component_distribution.support.check(value),
            component_log_probs,
            neg_inf_sentinel,
        )

    def log_prob(self, value: ArrayLike) -> jax.Array:
        comp_lp = self.component_log_probs(value)
        return logsumexp(
            jax.nn.log_softmax(self.mixing_distribution.logits)
            + comp_lp.sum(axis=self._dim_dim),
            axis=self.mixture_dim,
        )

    def component_sample(
        self, key: jax.Array, sample_shape: tuple = ()
    ) -> jax.Array:
        return self.component_distribution.sample(
            key,
            sample_shape=sample_shape,  # + self.event_shape
        )

    # def sample_with_intermediates(
    #     self, key: jax.random.PRNGKey, sample_shape: tuple = ()
    # ) -> tuple:
    #     """
    #     A version of ``sample`` that also returns the sampled component indices

    #     Parameters
    #     ----------
    #     key
    #         The rng_key key to be used for the distribution.
    #     sample_shape
    #         The sample shape for the distribution.

    #     Returns
    #     -------
    #     samples
    #         The samples from the distribution.
    #     indices
    #         The indices of the sampled components.
    #     """
    #     key_comp, key_ind = jax.random.split(key)
    #     samples = self.component_sample(key_comp, sample_shape=sample_shape)

    #     # Sample selection indices from the categorical (shape will be sample_shape)
    #     indices = self.mixing_distribution.expand(
    #         sample_shape + self.batch_shape
    #     ).sample(key_ind)
    #     indices_expanded = indices.reshape(indices.shape + (1,))

    #     # Select samples according to indices samples from categorical
    #     samples_selected = jnp.take_along_axis(
    #         samples, indices=indices_expanded, axis=-2
    #     )
    #     samples_selected = jnp.squeeze(samples_selected, axis=-1)

    #     return samples_selected, indices

    # def sample(self, key, sample_shape=()):
    #     return self.sample_with_intermediates(key=key, sample_shape=sample_shape)[0]
