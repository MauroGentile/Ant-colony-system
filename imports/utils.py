
from AI2020.IO_manager.io_tsp import TSP_Instance_Creator
import pandas as pd
import numpy as np
import datetime
import os


def parameters_dict(ants_no, p_instance, seed, time_limit, cl_bool, ql0,
            problem_istances,
            results_folder = None,
            plot = False, 
            random = True, 
            ramp_min_ant = 10, 
            ramp = False,
            beta = 2,
            alpha = 0.1,
            ro = 0.1):
    # conveniently create a parameter dictionary which will be used as input to run a ACS search 
    parameters={}
    parameters["prob_no"]=p_instance
    parameters["instance"] = TSP_Instance_Creator("standard", problem_istances[p_instance])
    parameters["alpha"] = alpha
    parameters["ro"] = ro
    parameters["beta"] = beta
    parameters["seed"] = seed
    parameters["time_limit"] = time_limit
    parameters["cl_bool"] = cl_bool
    parameters["cl"] = 20 if cl_bool == "Y" else parameters["instance"].nPoints
    parameters["q0"] = {}
    if ql0=="1-13/n":
        parameters["q0"]["value"] = 1-13/(parameters["instance"].nPoints)
    else:
        parameters["q0"]["value"]=float(ql0)
    parameters["q0"]["label"] = ql0
    parameters["m"] = ants_no
    parameters["plot"] = plot
    parameters["random"] = random
    parameters["ramp_min_ant"] = ramp_min_ant
    parameters["ramp"] = ramp
    parameters["results_folder"] = results_folder
    return(parameters)


def create_folder_structure(date):
    # it creates "results" subfolder contatining a folder whose name is equal to 
    # the date and time the simulation has been started.
    # Such a date folder, contain 2 subfolders: "log" and "video"
    #folder are created only if they do not exists
    if not(os.path.exists("results")):
        os.makedirs("results")
    root_folder=os.path.join('results',date)
    if not(os.path.exists(root_folder)):
        os.makedirs(os.path.join(root_folder, os.path.join("log")))
        os.makedirs(os.path.join(root_folder, os.path.join("video")))

def create_root_folder_name():
    #create the string with current date and time and call create_folder_structure 
    now=datetime.datetime.now()
    date=now.strftime("%Y%m%d_%H%M%S")
    create_folder_structure(date)
    results_folder=os.path.join("results",date)
    return(results_folder)

def initialize_df():
    #initialize an empty pandas datafarme with specific columns names 
    return(pd.DataFrame(columns=['Problem', 'Time limit','CL','q0','seed','Best lenght found','Gap','Iterations done']))


def add_log_line(parameters, acs):
    #extract the information of a simulation from acs class and create a row to be added to the pandas dataframe containing all the reults 
    c_label = str(parameters["cl"]) if parameters["cl_bool"] == "Y" else "n.a." 
    return([parameters["prob_no"],
     parameters["time_limit"],
     c_label,
     parameters["q0"]["label"],
     parameters["seed"],
     int(acs.global_best_tour_length),np.round(acs.global_best_tour_length/acs.ic.best_sol-1,3),
     acs.iteration])