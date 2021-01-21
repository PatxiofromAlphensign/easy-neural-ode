from jax.experimental import optimizers
import jax

from lib.optimizers import exponential_decay
from latent_ode import ode_kwargs, init_model

def setup_optimizers():
	global x,y,z
	model =init_model(ode_kwargs)
	return model
	grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))
	lr_seq = exponential_decay(step_size=1e-2, decay_steps=1,

					    decay_rate=0.999,
					    lowest=1e-2 / 10)
	x,y,z = optimizers.adamax(step_size=lr_seq)

#testing of jax jit enclosure or decorators
@jax.jit
def order():
	return

def main():
	opt = setup_optimizers()
	return opt["forward"]

main()()

