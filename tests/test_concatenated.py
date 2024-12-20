import jax
import jax.numpy as jnp
import numpyro
import pytest
from numpyro import distributions as dist
from numpyro.infer import Predictive

from stream_membership.distributions import NormalSpline
from stream_membership.distributions.concatenated import (
    ConcatenatedConstraints,
    ConcatenatedDistributions,
    _transform_to_concatenated,
)


class BaseTestConcatenated:
    values_expected_shape: tuple[tuple[jax.Array, tuple], ...] = (
        (jnp.array([0, 1, 2.0]), ()),
        (jnp.array([[0.0, 1, 2.0], [1.0, 2, 3.0], [2.0, 3, 4.0]]), (3,)),
    )
    sample_shapes = ((), (1,), (4,), (4, 5))

    def setup_dist(self):
        return NotImplementedError()

    def test_shape(self):
        x = self.setup_dist()
        assert x.event_shape == (3,)

    @pytest.mark.parametrize(("value", "expected_shape"), values_expected_shape)
    def test_logprob(self, value, expected_shape):
        x = self.setup_dist()
        assert x.log_prob(value).shape == expected_shape

    @pytest.mark.parametrize("sample_shape", sample_shapes)
    def test_sample(self, sample_shape):
        x = self.setup_dist()

        samples = x.sample(jax.random.PRNGKey(0), sample_shape=sample_shape)
        assert samples.shape == (*sample_shape, 3)

    def test_numpyro_predictive(self):
        def model():
            x = self.setup_dist()
            numpyro.sample("x", x)

        pred = Predictive(model, num_samples=10)(jax.random.PRNGKey(42))
        assert pred["x"].shape == (10, 3)


class TestAllUnivariate(BaseTestConcatenated):
    def setup_dist(self):
        x1 = dist.Normal(0, 1)
        x2 = dist.Normal(2.0, 0.5)
        x3 = dist.Normal(1.0, 0.25)
        return ConcatenatedDistributions([x1, x2, x3])


class TestUnivariateMultivariate(BaseTestConcatenated):
    def setup_dist(self):
        x1 = dist.Normal(0, 1)
        x2 = dist.MultivariateNormal(
            loc=jnp.array([1.0, 2.0]),
            covariance_matrix=jnp.array([[1.0, 0.0], [0, 0.5]]) ** 2,
        )
        return ConcatenatedDistributions([x1, x2])


class TestUnivariateSpline:
    x = jnp.arange(0, 10.0, 1.0)
    values_expected_shape = (
        (jax.random.normal(jax.random.PRNGKey(111), (x.size, 5)), x.shape),
    )
    sample_shapes = ((), (1,), (4,))

    def setup_dist(self):
        x1 = dist.Normal(0, 1)
        x2 = NormalSpline(
            knots=jnp.linspace(0, 10.0, 16),
            loc_vals=jnp.zeros(16),
            scale_vals=jnp.ones(16),
            x=self.x,
        )
        x3 = dist.Uniform(0, 1)
        x4 = dist.MultivariateNormal(
            loc=jnp.array([1.0, 2.0]),
            covariance_matrix=jnp.array([[1.0, 0.6], [0.6, 0.5]]) ** 2,
        )
        return ConcatenatedDistributions([x1, x2, x3, x4])

    def test_shape(self):
        x = self.setup_dist()
        assert x.event_shape == (5,)

    @pytest.mark.parametrize(("value", "expected_shape"), values_expected_shape)
    def test_logprob(self, value, expected_shape):
        x = self.setup_dist()
        assert x.log_prob(value).shape == expected_shape

    @pytest.mark.parametrize("sample_shape", sample_shapes)
    def test_sample(self, sample_shape):
        x = self.setup_dist()

        samples = x.sample(jax.random.PRNGKey(0), sample_shape=sample_shape)
        assert samples.shape == (*sample_shape, self.x.size, 5)

    @pytest.mark.xfail
    def test_numpyro_predictive(self):
        # TODO: maybe this doesn't work...
        def model():
            x = self.setup_dist()
            numpyro.sample("x", x)

        pred = Predictive(model, batch_ndims=1, num_samples=1)(jax.random.PRNGKey(42))
        assert pred["x"].shape == (self.x.size, 5)


class TestMixture(BaseTestConcatenated):
    def setup_dist(self):
        dist1 = TestAllUnivariate().setup_dist()
        dist2 = TestUnivariateMultivariate().setup_dist()
        mix = dist.CategoricalProbs(jnp.array([0.5, 0.5]))
        return dist.MixtureGeneral(mix, [dist1, dist2])


class TestConstraintsTransform:
    def setup_constraints(self):
        c1 = dist.constraints.positive
        c2 = dist.constraints.real
        c3 = dist.constraints.interval(0, 1)
        c4 = dist.constraints.interval(jnp.array([0.0, 2.0]), jnp.array([1.0, 100.0]))
        return ConcatenatedConstraints([c1, c2, c3, c4], sizes=[1, 2, 1, 2])

    def test_constraints(self):
        c = self.setup_constraints()

        test_data = jnp.full(6, 0.5)
        assert not c(test_data)

        test_data = jnp.array([1.0, 0.0, 0.0, 0.5, 1.0, 10.0])
        assert c(test_data)

        test_data = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.5, 1.0, 10.0],
                [-1.0, 0.0, 0.0, 0.5, 1.0, 10.0],
            ]
        )
        assert all(c(test_data) == jnp.array([True, False]))

    def test_transforms(self):
        c = self.setup_constraints()
        trans = _transform_to_concatenated(c)

        test_data = jnp.full(6, 0.5)
        assert trans(test_data).shape == test_data.shape
        assert jnp.allclose(test_data, trans.inv(trans(test_data)))

        test_data = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.5, 1.0, 10.0],
                [-1.0, 0.0, 0.0, 0.5, 1.0, 10.0],
            ]
        )
        assert trans(test_data).shape == test_data.shape
        with jax.experimental.enable_x64():
            assert jnp.allclose(
                test_data, trans.inv(trans(jnp.array(test_data, dtype=jnp.float64)))
            )
