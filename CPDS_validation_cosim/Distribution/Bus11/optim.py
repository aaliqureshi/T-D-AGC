import pyomo.environ as pe
import pyomo.opt as po
import numpy as np
import opendssdirect as dss
from opendssdirect.utils import run_command

# obj function : min abs(P_AGC - sum(del_P))

# s.t V_i + del_V_i <= V_max

num_DER = 5






# def LP(DER_headroom,del_agc,V_max,T_sens,Bus_voltage,DER_idx,DER_node_idx,DER_output,T_mat):
#     num_DER = len(DER_headroom)

#     solver = po.SolverFactory('glpk')

#     model = pe.ConcreteModel()

#     model.N = pe.RangeSet(1,num_DER)


