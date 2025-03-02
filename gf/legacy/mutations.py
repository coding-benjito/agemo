import itertools
import collections
import numpy as np
import sage.all
import gf.gflib as gflib

def single_partial(ordered_mutype_list, partial):
		return list(gflib.flatten(itertools.repeat(branchtype,count) for count, branchtype in zip(partial, ordered_mutype_list)))

def make_result_dict_from_mutype_tree_alt(gf, mutype_tree, theta, rate_dict, ordered_mutype_list, max_k, precision=165):
	root = tuple(0 for _ in max_k) #root is fixed
	num_mutypes = len(max_k)
	result = np.zeros(max_k + 2,dtype=object)
	stack = [(root, gf)]
	result[root] = eval_equation(gf, theta, rate_dict, root, precision)
	while stack:
		parent, parent_equation = stack.pop()
		if parent in mutype_tree:
			for child in mutype_tree[parent]:
				mucounts = [m for m, max_k_m in zip(child, max_k) if m<=max_k_m]
				marginal = len(mucounts)<num_mutypes
				child_equation = generate_equation(parent_equation, parent, child, max_k, ordered_mutype_list, marginal)
				stack.append((child, child_equation))
				result[child] = eval_equation(child_equation, theta, rate_dict, mucounts, precision)
	return result

def generate_equation(equation, parent, node, max_k, ordered_mutype_list, marginal):
	if marginal:
		marginals = {branchtype:0 for branchtype, count, max_k_m in zip(ordered_mutype_list, node, max_k) if count>max_k_m}
		return equation.subs(marginals)
	else:
		relative_config = [b-a for a,b in zip(parent, node)]
		partial = single_partial(ordered_mutype_list, relative_config)
		diff = sage.all.diff(equation, partial)
		return diff

def eval_equation(derivative, theta, ratedict, numeric_mucounts, precision):
	mucount_total = np.sum(numeric_mucounts)
	mucount_fact_prod = np.prod([np.math.factorial(count) for count in numeric_mucounts])
	#mucount_fact_prod = np.prod(factorials[numeric_mucounts])
	return sage.all.RealField(precision)((-1*theta)**(mucount_total)/mucount_fact_prod*derivative.subs(ratedict))

############ alternative make_result using depth first #####################
def depth_first_mutypes(max_k, labels, eq, theta, rate_dict, exclude=None, precision=165):
	#factorials = np.cumprod(np.arange(1, np.max(max_k)+1))
	#factorials = np.hstack((1,factorials))
	k = len(max_k) - 1
	stack = [(tuple([0 for _ in range(len(max_k))]), k, eq),]
	result = np.zeros(max_k+2, dtype=np.float64)
	if exclude is None:
		exclude = tuple()
	while stack:
		mutype, k, eq = stack.pop()
		if k>0:
			for step in single_step_df_mutypes_diff(mutype, labels[k], k, max_k[k], eq, theta, exclude):
				stack.append(step)
		else:
			for new_mutype, _, new_eq in single_step_df_mutypes_diff(mutype, labels[k], k, max_k[k], eq, theta, exclude):
				mucounts = np.array([m for m, max_k_m in zip(new_mutype, max_k) if m<=max_k_m])
				temp = eval_equation(new_eq, theta, rate_dict, mucounts, precision)
				#temp = eval_equation(new_eq, theta, rate_dict, mucounts, factorials, precision)
				result[new_mutype] = temp
				#result[mutype] = eval_equation(eq, theta, rate_dict, mucounts, precision)

	return result

def single_step_df_mutypes_diff(mutype, label, k, max_k, eq, theta, exclude):
	subsdict = {label:theta}
	# for i==0
	yield (mutype, k-1, eq.subs(subsdict))
	if len(exclude)==0 or (k!=exclude[0] or mutype[exclude[1]]==0):
		new_eq = eq
		#for i 1 .. max_k
		temp = list(mutype)
		for i in range(1, max_k+1):
			temp[k] = i
			new_eq = sage.all.diff(new_eq, label)
			yield (tuple(temp), k-1, new_eq.subs(subsdict))
		#for i==max_k+1
		subsdict[label] = 0
		temp[k] = max_k+1
		new_eq = eq.subs(subsdict)
		yield (tuple(temp), k-1, new_eq)

################ making branchtype dict #########################
def powerset(iterable):
	""" 
	returns generator containing all possible subsets of iterable
	"""
	s=list(iterable)
	return (''.join(sorted(subelement)) for subelement in (itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))))

def make_branchtype_dict(sample_list, mapping='unrooted', labels=None):
	"""
	Maps lineages to their respective mutation type
	Possible mappings: 'unrooted', 'label'
	"""
	all_branchtypes=list(gflib.flatten(sample_list))
	branches = [branchtype for branchtype in gflib.powerset(all_branchtypes) if len(branchtype)>0 and len(branchtype)<len(all_branchtypes)]	
	if mapping.startswith('label'):
		if labels:
			assert len(branches)==len(labels), "number of labels does not match number of branchtypes"
			branchtype_dict = {branchtype:sage.all.SR.var(label) for branchtype, label in zip(branches, labels)}
		else:
			branchtype_dict = {branchtype:sage.all.SR.var(f'z_{branchtype}') for branchtype in branches}
	elif mapping=='unrooted': #this needs to be extended to the general thing!
		if not labels:
			labels = ['m_1', 'm_2', 'm_3', 'm_4']
		assert set(all_branchtypes)=={'a', 'b'}
		branchtype_dict=dict()
		for branchtype in gflib.powerset(all_branchtypes):
			if len(branchtype)==0 or len(branchtype)==len(all_branchtypes):
				pass
			elif branchtype in ('abb', 'a'):
				branchtype_dict[branchtype] = sage.all.SR.var(labels[1]) #hetA
			elif branchtype in ('aab', 'b'):
				branchtype_dict[branchtype] = sage.all.SR.var(labels[0]) #hetB
			elif branchtype == 'ab':
				branchtype_dict[branchtype] = sage.all.SR.var(labels[2]) #hetAB
			else:
				branchtype_dict[branchtype] = sage.all.SR.var(labels[3]) #fixed difference
	else:
		ValueError("This branchtype mapping has not been implemented yet.")
	return branchtype_dict