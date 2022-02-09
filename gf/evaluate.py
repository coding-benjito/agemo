import numpy as np
import sys

import gf.gflib as gflib
import gf.diff as gfdiff
import gf.mutations as gfmuts

############################ getting out bSFS ########################################

class gfEvaluator:
	def __init__(self, gfobj, k_max, mutype_array):
		delta_idx = gfobj.exodus_rate #what value if None
		#only works with single delta_idx!
		self.eq_graph_array, eq_array, to_invert, eq_matrix = gfobj.equations_graph()		
		self.dependency_sequence = gfdiff.resolve_dependencies(self.eq_graph_array)
		self.num_branchtypes = len(k_max)
		self.final_result_shape = k_max+2
		size = len(mutype_array)	 
		#marg_iterator = gfdiff.marginals_nuissance_objects(k_max)
		#slices = marg_iterator[-1]
		#prepare_graph_evaluation_with_marginals_alt(eq_matrix, to_invert_array, 
		#eq_array, size, delta_idx, subsetdict, mutype_array, mutype_shape)
		self.subsetdict = gfdiff.product_subsetdict_marg_alt(tuple(self.final_result_shape), mutype_array)
		f_tuple = gfdiff.prepare_graph_evaluation_with_marginals_alt(
			eq_matrix, 
			to_invert, 
			eq_array,
			size,
			delta_idx,
			self.subsetdict,
			mutype_array,
			self.final_result_shape
			)
		num_eq_non_inverted = np.sum(to_invert==0) 
		num_eq_tuple = (num_eq_non_inverted, to_invert.size - num_eq_non_inverted)
		self.evaluator = gfdiff.evaluate_single_point_with_marginals_alt(
			size,
			num_eq_tuple,
			f_tuple
			)
		self.multiplier_matrix = gfdiff.taylor_to_probability_coeffs_alt(mutype_array, self.final_result_shape, include_marginals=True)

	def evaluate(self, theta, var, time):
		try:
			results = self.evaluator(var, time)
		except ZeroDivisionError:
			var = self.adjust_parameters(var)
			results = self.evaluator(var, time)
		final_result_flat = gfdiff.iterate_eq_graph_alt(self.dependency_sequence, self.eq_graph_array, results, self.subsetdict)
		theta_multiplier_matrix = gfdiff.taylor_to_probability(self.multiplier_matrix, theta)
		no_marginals = (theta_multiplier_matrix * final_result_flat).reshape(self.final_result_shape)
		
		#filling of array needed here!!!
		#final_result = np.zeros(self.final_result_shape, dtype=np.float64)
		
		return gfmuts.adjust_marginals_array(no_marginals, self.num_branchtypes)

	def adjust_parameters(self, var, factor=1e-5):
		epsilon = np.random.randint(low=-100, high=100, size=len(var) - self.num_branchtypes) * factor
		var[:-self.num_branchtypes]+=epsilon
		return var

class gfEvaluatorBT:
	def __init__(self, gfobj, k_max, mutype_array, ravel_mutype_array=None, subsetdict=None):
		delta_idx = gfobj.exodus_rate #what value if None
		#only works with single delta_idx!
		self.eq_graph_array, eq_array, to_invert, eq_matrix = gfobj.equations_graph()		
		self.dependency_sequence = gfdiff.resolve_dependencies(self.eq_graph_array)
		self.num_branchtypes = len(k_max)
		self.final_result_shape = k_max+2
		size = len(mutype_array)	 
		if subsetdict is None:
			self.subsetdict = gfdiff.product_subsetdict_marg_alt(tuple(self.final_result_shape), mutype_array)
		else self.subsetdict = subsetdict
		self.ravel_mutype_array = ravel_mutype_array #if None perform simple reshape
		f_tuple = gfdiff.prepare_graph_evaluation_with_marginals_alt(
			eq_matrix, 
			to_invert, 
			eq_array,
			size,
			delta_idx,
			self.subsetdict,
			mutype_array,
			self.final_result_shape
			)
		num_eq_non_inverted = np.sum(to_invert==0) 
		num_eq_tuple = (num_eq_non_inverted, to_invert.size - num_eq_non_inverted)
		self.evaluator = gfdiff.evaluate_single_point_with_marginals_alt(
			size,
			num_eq_tuple,
			f_tuple
			)
		self.multiplier_matrix = gfdiff.taylor_to_probability_coeffs_alt(mutype_array, self.final_result_shape, include_marginals=True)

	def evaluate(self, theta, var, time):
		try:
			results = self.evaluator(var, time)
		except ZeroDivisionError:
			var = self.adjust_parameters(var)
			results = self.evaluator(var, time)
		final_result_flat = gfdiff.iterate_eq_graph_alt(self.dependency_sequence, self.eq_graph_array, results, self.subsetdict)
		theta_multiplier_matrix = gfdiff.taylor_to_probability(self.multiplier_matrix, theta)
		no_marginals = (theta_multiplier_matrix * final_result_flat)
		
		if ravel_mutype_array not is None:
			final_result = np.zeros(self.final_result_shape, dtype=np.float64)
			final_result.flat[ravel_mutype_array] = final_result_flat
		else:
			final_result = final_result_flat.reshape(self.final_result_shape)

		return gfmuts.adjust_marginals_array(no_marginals, self.num_branchtypes)

	def adjust_parameters(self, var, factor=1e-5):
		epsilon = np.random.randint(low=-100, high=100, size=len(var) - self.num_branchtypes) * factor
		var[:-self.num_branchtypes]+=epsilon
		return var

def ravel_mutype_array(mutype_array, mutype_shape):
	size = mutype_array.shape[0]
	result = np.zeros(size, dtype=np.int64)
	mutype_shape = mutype_shape.astype(np.int64)
	mutype_array = mutype_array.astype(np.int64)
	for idx in range(size):
		result[idx] = gfdiff.ravel_multi_index(mutype_array[idx], mutype_shape)
	return result