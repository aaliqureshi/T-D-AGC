import numpy as np
import pyomo.environ as pe
import pyomo.opt as po
import set_DSS
import os
from pulp import *

# time_start = time_ns()
dir_to_feeder = os.getcwd()
num_DER = 100
method='direct'
feeder_type = '8500'
del_agc = 300 # unit is kW
DER_out_val = 10 # originally it was set as 500
V_max = 1.05
DER_output = [DER_out_val for idx in range(num_DER)]
# DER_output = [50,50,100,50]
DER_pert=[DER_output[idx]*1.05 for idx in range(num_DER)]
DER_max = [DER_output[idx]*2.5 for idx in range(num_DER)]
DER_headroom = [DER_output[idx]*2 for idx in range(num_DER)]
# DER_headroom = [125,125,125,125]



# DER_idx,Bus_voltage,initial_net_power=set_DSS.setDSS(num_DER,DER_output,del_agc) # call for 34 node feeder

DER_idx,DER_node_idx,Bus_voltage,initial_net_power=set_DSS.setDSS_8500_balanced(num_DER,DER_output,del_agc,feeder_type)


# solver = po.SolverFactory('glpk') # Linear Programming Kit
# model = pe.ConcreteModel()

prob = LpProblem('AGC Distribution Difference', LpMinimize)

LpVariable("")
