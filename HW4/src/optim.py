import numpy as np
import theano
import theano.tensor as TT
from nn_utils import *

class Optimizer(object):

	def __init__(self, name=None):
		self.name = name


class SGD(Optimizer):

	def __init__(self, inputs, costs, params, learning_rate=.1, clipping=1., name=None):
		self.name = name
		self.params = params
		self.grads = [init_zeros(p.get_value().shape) for p in params]

		gradients = TT.grad(costs, params)
		grads_clip, grads_norm = clip(gradients, clipping, square=False, params=self.params)

		grads_upd = [(grads, new_grads) for grads, new_grads in zip(self.grads, grads_clip)]

		self.update_grads = theano.function(inputs, [costs, TT.sqrt(grads_norm)], updates=grads_upd)

		lr = np.float32(learning_rate)
		delta = [lr*grads for grads in self.grads]
		params_upd = [(p, p-d) for p, d in zip(self.params, delta)]

		self.update_params = theano.function([], [], updates=params_upd)


class AdaDelta(Optimizer):

	def __init__(self, inputs, costs, params, gamma=0.75, eps=1e-6, clipping=1., name=None):
		super(AdaDelta, self).__init__()
		self.name = name
		self.params = params
		self.grads = [init_zeros(p.get_value().shape) for p in params]
		self.grads_sqr_avg = [init_zeros(p.get_value().shape) for p in params]
		self.delta_sqr_avg = [init_zeros(p.get_value().shape) for p in params]

		gradients = TT.grad(costs, self.params)
		grads_clip, grads_norm = clip(gradients, clipping, params=self.params)

		grads_upd = [(grads, new_grads) for grads, new_grads in zip(self.grads, grads_clip)]
		grads_sqr_avg_upd = [(grads_sqr_avg, gamma*grads_sqr_avg + (1.-gamma)*(new_grads**2.))
								for grads_sqr_avg, new_grads in zip(self.grads_sqr_avg, grads_clip) ]
		self.update_grads = theano.function(inputs, [costs, grads_norm], updates = grads_upd + grads_sqr_avg_upd)

		delta = [ grads * TT.sqrt(delta_sqr_avg+eps) / TT.sqrt(grads_sqr_avg+eps)
				for grads, delta_sqr_avg, grads_sqr_avg in zip(self.grads, self.delta_sqr_avg, self.grads_sqr_avg) ]

		delta_sqr_avg_upd = [ (delta_sqr_avg, gamma*delta_sqr_avg + (1.-gamma)*(new_delta**2.))
							for delta_sqr_avg, new_delta in zip(self.delta_sqr_avg, delta) ]
		params_upd = [(p,p-d) for p, d in zip(self.params, delta)]

		self.update_params = theano.function([], [], updates = params_upd + delta_sqr_avg_upd)


class Adam(Optimizer):

	def __init__(self, inputs, costs, params, clipping=1., alpha=0.005, beta_1=0.9, beta_2=0.999, eps=1e-8, name=None):
		super(Adam, self).__init__()
		self.name = name
		self.params = params

		self.grads = [init_zeros(p.get_value().shape) for p in params]
		self.m = [init_zeros(p.get_value().shape) for p in params]
		self.v = [init_zeros(p.get_value().shape) for p in params]
		self.beta1t = theano.shared(np.float32(beta_1))
		self.beta2t = theano.shared(np.float32(beta_2))

		gradients = TT.grad(costs, self.params)
		grads_clip, grad_norm = clip(gradients, clipping, params=self.params)

		update_ab = [(self.beta1t, self.beta1t * beta_1),
					(self.beta2t, self.beta2t * beta_2)]
		update_gc = [(gc, gr) for gc, gr in zip(self.grads, grads_clip)]
		m_up = [(m, beta_1 * m + (1. - beta_1) * gr) for m, gr in zip(self.m, grads_clip)]
		v_up = [(v, beta_2 * v + (1. - beta_2) * (gr ** 2)) for v, gr in zip(self.v, grads_clip)]

		self.update_grads = theano.function(inputs, [costs, grad_norm], updates=update_gc + m_up + v_up)

		param_up = [(p, p - alpha * (m / (1. - self.beta1t)) / (TT.sqrt(v / (1. - self.beta2t)) + eps)) for p, m, v in zip(self.params, self.m, self.v)]

		self.update_params = theano.function([],[], updates = update_ab + param_up)
