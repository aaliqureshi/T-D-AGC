import numpy as np
import opendssdirect as dss
from opendssdirect.utils import run_command
from sympy import quo
import set_DSS
import jacobian
import agc_allocation as ac
import DSS_PF
import plotter
import matplotlib.pyplot as plt
# from time import time_ns
import os
import csv
# from optim import LP

# time_start = time_ns()
dir_to_feeder = os.getcwd()
num_DER = 100
method='direct'
feeder_type = '8500'
del_agc = 1500  # unit is kW
DER_out_val = 10 # originally it was set as 500
V_max = 1.0511
DER_output = [DER_out_val for idx in range(num_DER)]
# DER_output = [50,50,100,50]
DER_pert=[DER_output[idx]*1.05 for idx in range(num_DER)]

consistent_random_object = np.random.RandomState(123)
max_factor = consistent_random_object.randint(2,6,num_DER,dtype=int)
# headroom_factor = consistent_random_object.random(1)

DER_max = [DER_output[idx]*max_factor[idx] for idx in range(num_DER)]
DER_headroom = [DER_max[idx]*1 for idx in range(num_DER)]
# DER_headroom = [125,125,125,125]



# DER_idx,Bus_voltage,initial_net_power=set_DSS.setDSS(num_DER,DER_output,del_agc) # call for 34 node feeder

DER_idx,DER_node_idx,Bus_voltage,initial_net_power=set_DSS.setDSS_8500_balanced(num_DER,DER_output,del_agc,feeder_type)



# build jacobian

# T_mat,T_sens=jacobian.Tan(DER_idx,DER_node_idx,DER_pert,DER_output,del_agc)
star='%'
print(star*10)
# print('Tangent sensitivities are: %s'%T_sens)
print(star*10)

S_mat,S_sens=jacobian.Sec (DER_idx,DER_node_idx,DER_max,DER_output,del_agc)

print(star*10)
# print('Secant sensitivities are: %s'%S_sens)
print(star*10)


# time_start= time_ns()
# print(time_start)
# Bus_voltage_T,DER_output_T,avg_sol_time_T=ac.AGC_calculation(DER_headroom, del_agc,V_max,T_sens,Bus_voltage,DER_idx,DER_node_idx,DER_output,T_mat)
# time_end=time_ns()
# print(time_end)
# print(f'$$$$$$$$$$$$$$$ Average time to solve LPF is: {avg_sol_time_T} milli-sec.$$$$$$$$$$$$$$')

# print(f'AGC allocated in {time_end - time_start} sec.')
# DER_output = [DER_out_val for idx in range(num_DER)]

# Bus_voltage_S,DER_output_S,avg_sol_time_S,DER_output_LP=ac.AGC_calculation(DER_headroom, del_agc,V_max,S_sens,Bus_voltage,DER_idx,DER_node_idx,DER_output,S_mat)
Bus_voltage_S,DER_output_S,DER_output_LP=ac.AGC_calculation(DER_headroom, del_agc,V_max,S_sens,Bus_voltage,DER_idx,DER_node_idx,DER_output,S_mat)

print(f'======> DER Output (Proposed): {DER_output_S}')

print(f'======> DER Output (LP): {DER_output_LP}')

# print(f'$$$$$$$$$$$$$$$ Average time to solve LPF is: {avg_sol_time_S} milli-sec.$$$$$$$$$$$$$$')

# LP(DER_headroom,del_agc,V_max,T_sens,Bus_voltage,DER_idx,DER_node_idx,DER_output,T_mat)


# DER_output = [DER_out_val for idx in range(num_DER)]


# _,Bus_voltage_DSS_T,net_power_T=DSS_PF.solvePF_8500_balanced(DER_output_T,DER_idx,del_agc,typee=None,DER=None,store=0)

_,Bus_voltage_DSS_S,net_power_S=DSS_PF.solvePF_8500_balanced(DER_output_S,DER_idx,del_agc,typee=None,DER=None,store=0)

gp = sum([Bus_voltage_DSS_S[i]>V_max for i in range(len(Bus_voltage))])
print('Actual Voltage violations in proposed method are %s' %gp)

_,Bus_voltage_DSS_LP,net_power_LP=DSS_PF.solvePF_8500_balanced(DER_output_LP,DER_idx,del_agc,typee=None,DER=None,store=0)


gp = sum([Bus_voltage_DSS_LP[i]>V_max for i in range(len(Bus_voltage))])
print('OP Actual Voltage violations in proposed method are %s' %gp)


# import_diff_T,ratio_T= DSS_PF.response_ratio(initial_net_power,net_power_T,DER_output_T,del_agc)
import_diff_S,ratio_S= DSS_PF.response_ratio(initial_net_power,net_power_S,DER_output_S,del_agc)

import_diff_LP,ratio_LP= DSS_PF.response_ratio(initial_net_power,net_power_LP,DER_output_LP,del_agc)

# percent_error_T, T_error_max, T_error_avg  = DSS_PF.Percent_error (Bus_voltage_DSS_T, Bus_voltage_T,del_agc)
# print(f'Max. % error for tangent is: {T_error_max}')
# print(f'Avg. % error for tangent is: {T_error_avg}')
percent_error_S, S_error_max,S_error_avg = DSS_PF.Percent_error (Bus_voltage_DSS_S, Bus_voltage_S,del_agc)
# print(f'Max. % error for secant is: {S_error_max}')
# print(f'Avg. % error for secant is: {S_error_avg}')


# max_error_T = DSS_PF.Max_error (Bus_voltage_DSS_T, Bus_voltage_T,del_agc)
# max_error_S = DSS_PF.Max_error (Bus_voltage_DSS_S, Bus_voltage_S,del_agc)
# time_end = time_ns()

# print(time_end - time_start)

# print('~~~~~~~~Max. Error (tangent): %s' %max_error_T)

# print('~~~~~~~~Max. Error (secant) is %s' %max_error_S)
# plotter.plotty (percent_error_S,DER_output,DER_output_S,key='Secant')
# print('~~~~~~~~Percent. Error (tangent) is %s' %percent_error_T)
# plotter.plotty (percent_error_T,DER_output,DER_output_T,key='Tangent')

# plotter.plot_ratio(key='S')

# print('~~~~~~~~Percent. Error (secant) is %s' %percent_error_S) 
# plotty (percent_error_S,DER_output,DER_output_S,key='Secant')

# plt.figure(1)
# plt.plot(Bus_voltage_DSS_T)
# plt.show()

# filename = 'Tangent_error.txt'
# with open ('Tangent_error.txt','a') as file:
#     file.write(str(T_error_max))
# print(str(T_error_avg))
# print(float(T_error_avg))



#####################################################################################

## uncomment if need to store & plot results

# rows_T = [str(del_agc),str(T_error_avg),str(T_error_max)]
# rows_S = [str(del_agc),str(S_error_avg),str(S_error_max)]

# with open('Tangent_errors.csv','a',newline='') as csvfile:
#     error_writer = csv.writer(csvfile,quoting=csv.QUOTE_NONE)
#     error_writer.writerow(rows_T)

# with open('Secant_errors.csv','a',newline='') as csvfile:
#     error_writer = csv.writer(csvfile,quoting=csv.QUOTE_NONE)
#     error_writer.writerow(rows_S)

# # uncomment for plotting!!
# plotter.plot_ratio('Tangent',method)
# plotter.plot_ratio('Secant',method)

#########################################################################################