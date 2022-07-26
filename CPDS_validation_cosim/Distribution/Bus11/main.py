import os
import numpy as np
import opendssdirect as dss
from opendssdirect.utils import run_command
import set_DSS
import jacobian
import agc_allocation as ac
import DSS_PF



num_DER = 4
feeder_type = '13'
del_agc = 1050 # unit is kW
DER_out_val = 500
V_max = 1.02
DER_output = [DER_out_val for idx in range(num_DER)]
DER_pert=[DER_output[idx]*1.01 for idx in range(num_DER)]
DER_max = [DER_output[idx]*2 for idx in range(num_DER)]
DER_headroom = [DER_output[idx]*0.9 for idx in range(num_DER)]


v0,a0,DER_idx,Bus_voltage=set_DSS.setDSS(num_DER,DER_output,del_agc)

# build jacobian

T_mat,T_sens=jacobian.Tan(v0,a0,DER_idx,DER_pert,DER_output,del_agc)

S_mat,S_sens=jacobian.Sec (v0,a0,DER_idx,DER_max,DER_output,del_agc)

print('sensitivities are: '%T_sens)

Bus_voltage_T,DER_output_T=ac.AGC_calculation(DER_headroom, del_agc,V_max,T_sens,Bus_voltage,DER_idx,DER_output,T_mat)

Bus_voltage_S,DER_output_S=ac.AGC_calculation(DER_headroom, del_agc,V_max,S_sens,Bus_voltage,DER_idx,DER_output,S_mat)

_,Bus_voltage_DSS_T=DSS_PF.solvePF(DER_output_T,DER_idx,del_agc,type=None,DER=None,store=0)
_,Bus_voltage_DSS_S=DSS_PF.solvePF(DER_output_S,DER_idx,del_agc,type=None,DER=None,store=0)

percent_error_T = DSS_PF.Percent_error (Bus_voltage_DSS_T, Bus_voltage_T,del_agc)
percent_error_S = DSS_PF.Percent_error (Bus_voltage_DSS_S, Bus_voltage_S,del_agc)

max_error_T = DSS_PF.Max_error (Bus_voltage_DSS_T, Bus_voltage_T,del_agc)
max_error_S = DSS_PF.Max_error (Bus_voltage_DSS_S, Bus_voltage_S,del_agc)

print('~~~~~~~~Max. Error (tangent): %s' %max_error_T)
print('~~~~~~~~Max. Error (secant) is %s' %max_error_S)

print('~~~~~~~~Percent. Error (tangent) is %s' %percent_error_T)
print('~~~~~~~~Percent. Error (secant) is %s' %percent_error_S) 

