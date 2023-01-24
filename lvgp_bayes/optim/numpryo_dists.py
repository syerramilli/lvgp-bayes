import math
import jax.numpy as jnp
from jax.lax import broadcast_shapes,clamp
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes

class MollifiedUniform(dist.Distribution):
    arg_constraints = {'a':constraints.real,'b':constraints.real,'tail_sigma':constraints.positive}
    support = constraints.real
    def __init__(self,a,b,tail_sigma=0.1):
        self.a,self.b,self.tail_sigma = promote_shapes(a,b,tail_sigma)
        batch_shape = broadcast_shapes(jnp.shape(a),jnp.shape(b),jnp.shape(tail_sigma))
        super().__init__(batch_shape)

    @property
    def mean(self):
        return (self.a+self.b)/2
    
    @property
    def _half_range(self):
        return (self.b-self.a)/2

    @property
    def _log_normalization_constant(self):
        return -jnp.log(1+(self.b-self.a)/(math.sqrt(2*math.pi)*self.tail_sigma))

    def log_prob(self,X):
        # expression preserving gradients under automatic differentiation
        tail_dist = clamp(0.,jnp.abs(X-self.mean)-self._half_range,float('inf'))
        return dist.Normal(loc=jnp.zeros_like(self.a),scale=self.tail_sigma).log_prob(tail_dist)+self._log_normalization_constant
    
    def sample(self,key,sample_shape=()):
        return dist.Uniform(self.a,self.b).sample(key,sample_shape)