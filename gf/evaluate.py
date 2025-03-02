import numpy as np
import sys

import gf.gflib as gflib
import gf.diff as gfdiff
import gf.mutations as gfmuts

############################ getting out bSFS ########################################

class gfEvaluator:
	def __init__(self, gfobj, k_max):
		delta_idx = gfobj.exodus_rate #what value if None
		#only works with single delta_idx!
		self.eq_graph_array, eq_array, to_invert, eq_matrix = gfobj.equations_graph()		
		self.dependency_sequence = gfdiff.resolve_dependencies(self.eq_graph_array)
		self.num_mutypes = len(k_max)
		final_result_shape = k_max+2	
		marg_iterator = gfdiff.marginals_nuissance_objects(k_max)
		slices = marg_iterator[-1]
		f_array = gfdiff.prepare_graph_evaluation_with_marginals(
			eq_matrix, 
			to_invert, 
			eq_array,
			marg_iterator, 
			delta_idx
			)
		num_eq_non_inverted = np.sum(to_invert==0) 
		num_eq_tuple = (num_eq_non_inverted, to_invert.size - num_eq_non_inverted)
		self.evaluator = gfdiff.evaluate_single_point_with_marginals(
			k_max, 
			f_array,
			num_eq_tuple,
			slices
			)
		self.subsetdict = gfdiff.product_subsetdict_marg(tuple(final_result_shape))
		self.multiplier_matrix = gfdiff.taylor_to_probability_coeffs(k_max+1, include_marginals=True)

	def evaluate(self, theta, var, time):
		try:
			results = self.evaluator(var, time)
		except ZeroDivisionError:
			var = self.adjust_parameters(var)
			results = self.evaluator(var, time)
		final_result = gfdiff.iterate_eq_graph(self.dependency_sequence, self.eq_graph_array, results, self.subsetdict)
		theta_multiplier_matrix = gfdiff.taylor_to_probability(self.multiplier_matrix, theta)
		no_marginals = theta_multiplier_matrix * final_result
		return gfmuts.adjust_marginals_array(no_marginals, self.num_mutypes)

	def adjust_parameters(self, var, factor=1e-5):
		epsilon = np.random.randint(low=-100, high=100, size=len(var) - self.num_mutypes) * factor
		var[:-self.num_mutypes]+=epsilon
		return var