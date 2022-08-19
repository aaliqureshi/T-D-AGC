import opendssdirect as dss
from opendssdirect.utils import run_command
import numpy as np
import os
import pandas as pd
from set_DSS import dir_to_feeder
from time import time_ns


# dir_to_feeder=os.getcwd()
# print('<><><><><>I am in %s'%dir_to_feeder)

def solvePF (DER_output, index_DERs,del_agc,typee,DER,store):
    """Solve power flow using OpenDSS based on new DER_output
    returns total number of nodes in the system & bus voltage magnitude, save node bus voltages in a csv file if
    input argument store!=0
    Re-create the case whith new parameters to avoid any data over-ride."""

    print('*'*10)
    print('Received DER output is %s' %DER_output)
    os.chdir(dir_to_feeder)

    # dir_to_feeder = os.getcwd()
    # print(dir_to_feeder)
    # run_command('clear')
    dss.Text.Command('clear')
    print(dss.Text.Result())
    # file_name=r"C:\Users\aaliq\Documents\AGC Codes\T-D-AGC\CPDS_validation_cosim\Distribution\Bus11\IEEE34Bus\ieee34_balanced.dss"

    # run_command ('compile IEEE34Bus/ieee34_test.dss')
    run_command ('compile IEEE8500/Master.dss')

    # get load kw values
    BASEKV = dss.Vsources.BasekV()
    loadname_all=[]
    loadkw_all=[]
    loadkvar_all=[]
    num_loads = dss.Loads.Count()
    dss.Loads.First() # Set first Load active; returns 0 if none.
    for i in range(num_loads):
        loadname = dss.Loads.Name() # Get/set the name of the active Load
        #print(f'loadname={loadname}')
        loadkw = dss.Loads.kW() # Set kW for active Load. Updates kvar based on present PF.
        loadkvar = dss.Loads.kvar() # Get/set kvar for active Load. If set, updates PF based on present kW.
        loadname_all.append(loadname)
        loadkw_all.append(loadkw)
        loadkvar_all.append(loadkvar)
        dss.Loads.Next() # Sets next Load active; returns 0 if no more

    # loadkw_dict = dict(zip(loadname_all, loadkw_all))
    # loadkvar_dict = dict(zip(loadname_all, loadkvar_all))
    allnodenames = dss.Circuit.AllNodeNames()
    num_nodes = len(allnodenames)
    
    print(f"Total base line power demand (3-phase) before DER connection is {dss.Circuit.TotalPower()}")
    

    i=1
    # index_DERs = [35,31,41,91]
    for ind in index_DERs:
        nodename = allnodenames[ind]
        # print(nodename)
        dss.Circuit.SetActiveBus(nodename)
        buskv = dss.Bus.kVBase()
        # print(nodename)
        # print(buskv)
        # print(DER_output[i-1])
        run_command(f'New generator.der_{i} bus1={nodename} Phases=1 Conn=Wye Model=1 kV={buskv} kw={DER_output[i-1]} kvar=0')
        i=i+1

    
    time_start=time_ns()
    dss.Text.Command('solve')
    time_end=time_ns()
    print(f'DSS solved executed in {time_end-time_start} sec')
    # res=dss.Text.Result()
    # print(res)
    print(f'Network active power demand after DER connection is{dss.Circuit.TotalPower()}')

    dss.Generators.First()
    for i in range (4):
        print(f'DER {i} was set to {dss.Generators.kW()}')
        dss.Generators.Next()
    print('*'*10)

    allbusmagpu = dss.Circuit.AllBusMagPu()
    net_power = dss.Circuit.TotalPower()

    allbusmagpu_base = np.array(allbusmagpu)

    if store!=0:
        voltage_results = pd.DataFrame(allbusmagpu_base)
        dir_to_results = os.path.join(dir_to_feeder, "simulation_results")
        voltage_results.to_csv(dir_to_results+'\\'+typee+'_results_'+str(del_agc)+'_'+str(DER)+'.csv')
    
    run_command('clear')
    dss.Basic.ClearAll()
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@{}'.format(dss.Bus.kVBase()))

    return num_nodes,allbusmagpu,-net_power[0]



def solvePF_8500_balanced (DER_output, index_DERs,del_agc,typee,DER,store):

    """Solve power flow using OpenDSS based on new DER_output
    returns total number of nodes in the system & bus voltage magnitude, save node bus voltages in a csv file if
    input argument store!=0
    Re-create the case whith new parameters to avoid any data over-ride."""

    # print('*'*10)
    # print('Received DER output is %s' %DER_output)
    os.chdir(dir_to_feeder)

    # dir_to_feeder = os.getcwd()
    # print(dir_to_feeder)
    # run_command('clear')
    dss.Text.Command('clear')
    # print(dss.Text.Result())
    # file_name=r"C:\Users\aaliq\Documents\AGC Codes\T-D-AGC\CPDS_validation_cosim\Distribution\Bus11\IEEE34Bus\ieee34_balanced.dss"

    # run_command ('compile IEEE34Bus/ieee34_test.dss')
    run_command ('compile IEEE8500/Master.dss')

    # get load kw values
    # BASEKV = dss.Vsources.BasekV()
    # loadname_all=[]
    # loadkw_all=[]
    # loadkvar_all=[]
    # num_loads = dss.Loads.Count()
    # dss.Loads.First() # Set first Load active; returns 0 if none.
    # for i in range(num_loads):
    #     loadname = dss.Loads.Name() # Get/set the name of the active Load
    #     #print(f'loadname={loadname}')
    #     loadkw = dss.Loads.kW() # Set kW for active Load. Updates kvar based on present PF.
    #     loadkvar = dss.Loads.kvar() # Get/set kvar for active Load. If set, updates PF based on present kW.
    #     loadname_all.append(loadname)
    #     loadkw_all.append(loadkw)
    #     loadkvar_all.append(loadkvar)
    #     dss.Loads.Next() # Sets next Load active; returns 0 if no more

    # loadkw_dict = dict(zip(loadname_all, loadkw_all))
    # loadkvar_dict = dict(zip(loadname_all, loadkvar_all))
    allnodes = dss.Circuit.AllNodeNames()
    num_nodes= len(allnodes)
    # num_nodes = len(allnodenames)
    
    # print(f"Total base line power demand (3-phase) before DER connection is {dss.Circuit.TotalPower()}")

    num_loads = dss.Loads.Count()
    dss.Loads.First()
    loadbuses_two_phase = list()

    for idx in range(num_loads):
        # print(dss.Loads.Name())
        # print(dss.CktElement.BusNames())
        loadbuses_two_phase.append(dss.CktElement.BusNames())
        # print(dss.CktElement.NodeNames())
        dss.Loads.Next()

    
    j=0
    for idx in range (len(index_DERs)):
        run_command(f'New generator.der_{idx+1} bus1={loadbuses_two_phase[index_DERs[idx]][0]} phases =2 model =1 conn= wye kV=0.208 kW={DER_output[idx]} kvar = 0')
        # print(f'DER {idx+1} connected to bus {loadbuses_two_phase[index_DERs[idx]][0]}')
        # print(f'Corresponding nodes are: {allnodes[DER_index[j]]} and {allnodes[DER_index[j+1]]} ')
        j+=2


    # i=1
    # # index_DERs = [35,31,41,91]
    # for ind in index_DERs:
    #     nodename = allnodenames[ind]
    #     # print(nodename)
    #     dss.Circuit.SetActiveBus(nodename)
    #     buskv = dss.Bus.kVBase()
    #     # print(nodename)
    #     # print(buskv)
    #     # print(DER_output[i-1])
    #     run_command(f'New generator.der_{i} bus1={nodename} Phases=1 Conn=Wye Model=1 kV={buskv} kw={DER_output[i-1]} kvar=0')
    #     i=i+1

    
    time_start=time_ns()
    dss.Text.Command('solve')
    time_end=time_ns()
    # print(f'DSS solved executed in {time_end-time_start} sec')
    total_time = (time_end - time_start)/1e6
    print(f'$$$$$$$$$$$$ DSS solution time is {total_time} sec.')
    # res=dss.Text.Result()
    # print(res)
    # print(f'Network active power demand after DER connection is{dss.Circuit.TotalPower()}')

    # dss.Generators.First()
    # for i in range (4):
    #     print(f'DER {i} was set to {dss.Generators.kW()}')
    #     dss.Generators.Next()
    print('*'*10)

    allbusmagpu = dss.Circuit.AllBusMagPu()
    net_power = dss.Circuit.TotalPower()

    allbusmagpu_base = np.array(allbusmagpu)

    if store!=0:
        voltage_results = pd.DataFrame(allbusmagpu_base)
        dir_to_results = os.path.join(dir_to_feeder, "simulation_results")
        voltage_results.to_csv(dir_to_results+'\\'+typee+'_results_'+str(del_agc)+'_'+str(DER)+'.csv')
    
    run_command('clear')
    dss.Basic.ClearAll()
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@{}'.format(dss.Bus.kVBase()))

    return num_nodes,allbusmagpu,-net_power[0]






def Percent_error (Bus_voltage_DSS, Bus_voltage_LPF,del_agc):
    """find percent error for each node in the system using voltage calculated from OpenDSS & out method"""
    error = [abs(((Bus_voltage_DSS[idx] - Bus_voltage_LPF[idx]))/Bus_voltage_DSS[idx])*100 for idx in range(len(Bus_voltage_DSS))]
    error_max = max(error)
    error_avg = sum(error)/len(error)

    # dir=os.getcwd()
    # os.path.join(dir,'Error Results')
    return error,error_max,error_avg

def Max_error (Bus_voltage_DSS, Bus_voltage_LPF,del_agc):
    """find max abs. error in the system using voltage calculated from OpenDSS & out method"""
    error = [abs(Bus_voltage_DSS[idx] - Bus_voltage_LPF[idx]) for idx in range(len(Bus_voltage_DSS))]
    # dir=os.getcwd()
    # os.path.join(dir,'Error Results')
    return max(error)

def response_ratio (initial_power,power,DER_output,agc_request):
    power_import_diff = initial_power - power
    ratio = power_import_diff / agc_request

    print('*%*^*&*')
    print(f'AGC request is {agc_request} kW.')
    print(f'Initial power is {initial_power} kW.')
    print(f'Final power is {power} kW.')

    print(f'Difference in power import is: {power_import_diff}')
    print(f'Response ratio is {ratio}')
    print('*%*^*&*')

    return power_import_diff, ratio