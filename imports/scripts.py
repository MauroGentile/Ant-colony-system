import os
import pandas as pd
from imports.ACS import ACS
from imports.utils import *
import pickle

def run_simulations(problems,seeds,time_limits,cl_bools,qs,problem_istances):
	results_folder = create_root_folder_name()
	df = initialize_df()
	best_tours = {}
	for p in problems:
	    for s in seeds:
	        for t in time_limits:
	            for c in cl_bools:
	                for q in qs:
	                    parameters = parameters_dict(ants_no = 10,
	                           p_instance = p,
	                           seed = s,
	                           time_limit = t,
	                           cl_bool = c,
	                           ql0 = q,
	                           problem_istances = problem_istances,
	                           results_folder = results_folder)
	                    acs = ACS(parameters)
	                    acs.run()
	                    name = 'problem: ' + str(p) + ' seed: ' + str(s) + ' time: ' + str(t) + ' cl: ' + str(c) + ' q: ' + q
	                    best_tours[name] = acs.best_tours_length
	                    df.loc[df.shape[0]] = add_log_line(parameters, acs)
	
	fn=os.path.join(parameters["results_folder"], os.path.join("log","data"))
	f = open(fn+".pkl","wb")
	pickle.dump(best_tours,f)
	f.close()
	df.to_csv(fn+".csv", index=False) 
	return df