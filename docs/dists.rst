Distributions
=============

If you would like to add a new distribution, please open a pull request.

List of distributions
---------------------

:class:`NormalSpline`:
Represents a Normal distribution where the loc and (log)scale parameters are
controlled by splines that are evaluated at some other parameter values x. In
other words, this distribution is conditional on x.

:class:`TruncatedNormalSpline`:
Equivalent to :class:`NormalSpline`, but for a truncated Normal distribution

:class:`Normal1DSplineMixture`:
Represents a mixture of Normal distributions where the loc and (log)scale parameters are
controlled by splines that are evaluated at some other parameter values x. Takes either a
``mixing_distribution`` parameter (a single, fixed weighting of the mixture components, shared
across all x) or a ``mixing_vals`` parameter (per-knot mixing logits for each component, shape
``(n_components, n_knots)``, exactly like ``loc_vals``/``scale_vals``): when ``mixing_vals`` is
given, the mixing weights are themselves a smooth spline function of x instead of fixed. Exactly
one of ``mixing_distribution`` or ``mixing_vals`` must be provided.

:class:`TruncatedNormal1DSplineMixture`:
Equivalent to :class:`Normal1DSplineMixture`, but for multiple truncated Normal distributions

:class:`IndependentGMM`:
Represents a Gaussian Mixture Model where the components are fixed to their input locations
and there are no covariances (but each dimension can have different scales / standard deviations).

:class:`DirichletSpline`:
Represents a Dirichlet distribution where the concentration parameters are
controlled by splines that are evaluated at some other parameter values x.

:class:`ConcatenatedDistributions`:
Represents a multi-dimensional distribution that is the concatenation of multiple distributions.

:class:`FlowDensity`:
Wraps a (typically pretrained) ``flowjax`` normalizing flow as a numpyro distribution for one
or more jointly-modeled coordinates (e.g. a color-magnitude density). Like
:class:`NormalSpline`, this can optionally be conditioned on some other coordinate x (e.g.
``phi1``). The flow's internal parameters are not exposed to numpyro and are not updated
during inference -- it is meant to be trained separately and loaded in as a frozen density.
