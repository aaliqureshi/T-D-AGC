import opendssdirect as dss
from opendssdirect.utils import run_command
import numpy as np
import os
import pandas as pd


dir_to_feeder=os.getcwd()
print('<><><><><>I am in %s'%dir_to_feeder)

def solvePF (DER_output, index_DERs,del_agc,type,DER,store):
    """Solve power flow based using OpenDSS based on new DER_output
    returns total number of nodes in the system & bus voltage magnitude, save node bus voltages in a csv file if
    input argument store!=0"""

    print('Received DER output is %s' %DER_output)

    allnodenames = dss.Circuit.AllNodeNames()
    num_nodes = len(allnodenames)

    j=1
    for ind in index_DERs:
        nodename = allnodenames[ind]
        run_command(f'Edit generator.der_{j} kw={DER_output[j-1]/3}') # opendss positive sequence should defined as 1 phase
        j=j+1

    dss.run_command("Solve number=1 stepsize=1s")

    allbusmagpu = dss.Circuit.AllBusMagPu()

    S = dss.Circuit.TotalPower() # the total power is negative
    S_net=dss.Circuit.TotalPower()

    S[0] = S[0] - np.sum(DER_output) # adjust
    P = -S[0]
    #
    print(f"======Total power demand = {dss.Circuit.TotalPower()}")
    print('Net active power is: %s' %P)
    allbusmagpu_base = np.array(allbusmagpu)

    if store!=0:
        voltage_results = pd.DataFrame(allbusmagpu_base)
        dir_to_results = os.path.join(dir_to_feeder, "simulation_results")
        voltage_results.to_csv(dir_to_results+'\\'+type+'_results_'+str(del_agc)+'_'+str(DER)+'.csv')

    return num_nodes,allbusmagpu


def Percent_error (Bus_voltage_DSS, Bus_voltage_LPF,del_agc):
    """find percent error for each node in the system using voltage calculated from OpenDSS & out method"""
    error = [((Bus_voltage_DSS[idx] - Bus_voltage_LPF[idx])/Bus_voltage_DSS[idx])*100 for idx in range(len(Bus_voltage_DSS))]
    # dir=os.getcwd()
    # os.path.join(dir,'Error Results')
    return error

def Max_error (Bus_voltage_DSS, Bus_voltage_LPF,del_agc):
    """find max abs. error in the system using voltage calculated from OpenDSS & out method"""
    error = [abs(Bus_voltage_DSS[idx] - Bus_voltage_LPF[idx]) for idx in range(len(Bus_voltage_DSS))]
    # dir=os.getcwd()
    # os.path.join(dir,'Error Results')
    return max(error)