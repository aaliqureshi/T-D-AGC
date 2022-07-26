import opendssdirect as dss
from opendssdirect.utils import run_command
import numpy as np
import os
import pandas as pd


dir_to_feeder=os.getcwd()
print('<><><><><>I am in %s'%dir_to_feeder)

def solvePF (DER_output, index_DERs,del_agc,type,DER,store):
    print('Received DER output is %s' %DER_output)
    # run_command('compile DSSfiles/Master.dss')
    # run_command("BatchEdit Load..* Vminpu=0.9")
    # run_command("New Loadshape.dummy npts=60 sinterval=1 Pmult=[file=dummy_profile.txt]")
    # run_command("BatchEdit Load..* Yearly=dummy")
    # run_command("set mode=snapshot")
    # dss.Solution.ControlMode(2)

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
    allnodenames = dss.Circuit.AllNodeNames()
    num_nodes = len(allnodenames)

    # load_mult = 1.0111
    # loadkw_new_all = []
    # dss.Loads.First()

    # for j in range(num_loads):
    #     loadname = dss.Loads.Name()
    #     # print(f"{loadname} kw={dss.Loads.kW()}")
    #     # text = run_command("? Load.p_perturb.enabled")

    #     # now calculate kw and kvar based on the csv
    #     #load_mult = load_profile_random[int(current_time)]
    #     loadkw_new = load_mult * loadkw_dict[loadname]
    #     loadkw_new_all.append(loadkw_new)
    #     loadkvar_new = load_mult * loadkvar_dict[loadname]
    #     # now change opendss load object
    #     run_command('edit load.{ln} kw={kw}'.format(ln=loadname, kw=loadkw_new))
    #     run_command('edit load.{ln} kvar={kw}'.format(ln=loadname, kw=loadkvar_new))
    #     # sum+=loadkw_new
    #     dss.Loads.Next()

    # dss.Vsources.PU(v0) # Per-unit value of source voltage
    # dss.Vsources.AngleDeg(np.degrees(a0)) # Phase angle of first phase in degrees

    # i=1
    # for ind in index_DERs:
    #     nodename = allnodenames[ind]
    #     # print(nodename)
    #     dss.Circuit.SetActiveBus(nodename)
    #     buskv = dss.Bus.kVBase()
    #     run_command(f'Edit generator.der_{i} bus1={nodename} Phases=1 Conn=Wye Model=1 kV={buskv} kw={DER_output[i-1]} kvar=0')
    #     i=i+1
    # run_command("BatchEdit Generator..* Yearly=dummy")

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
    error = [((Bus_voltage_DSS[idx] - Bus_voltage_LPF[idx])/Bus_voltage_DSS[idx])*100 for idx in range(len(Bus_voltage_DSS))]
    # dir=os.getcwd()
    # os.path.join(dir,'Error Results')
    return error

def Max_error (Bus_voltage_DSS, Bus_voltage_LPF,del_agc):
    error = [abs(Bus_voltage_DSS[idx] - Bus_voltage_LPF[idx]) for idx in range(len(Bus_voltage_DSS))]
    # dir=os.getcwd()
    # os.path.join(dir,'Error Results')
    return max(error)