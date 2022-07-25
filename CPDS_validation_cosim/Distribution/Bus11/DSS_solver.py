import opendssdirect as dss
from opendssdirect.utils import run_command
import numpy as np


def solveDSS(DER_output) :
    run_command('compile DSSfiles/Master.dss')
    run_command('BatchEdit Load..* Vminpu = 0.9')
    run_command("New Loadshape.dummy npts=60 sinterval=1 Pmult=[file=dummy_profile.txt]")
    run_command("BatchEdit Load..* Yearly=dummy")
    run_command("set mode=snapshot")
    dss.Solution.ControlMode(2)
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

    print('@@@@@@@@@@Sum of actual load is %s' %sum(loadkw_all))
    loadkw_dict = dict(zip(loadname_all, loadkw_all))
    loadkvar_dict = dict(zip(loadname_all, loadkvar_all))
    allnodenames = dss.Circuit.AllNodeNames()
    num_nodes = len(allnodenames)

    load_mult = 1.0111
    loadkw_new_all = []
    dss.Loads.First()

    for j in range(num_loads):
        loadname = dss.Loads.Name()
        # print(f"{loadname} kw={dss.Loads.kW()}")
        # text = run_command("? Load.p_perturb.enabled")

        # now calculate kw and kvar based on the csv
        #load_mult = load_profile_random[int(current_time)]
        loadkw_new = load_mult * loadkw_dict[loadname]
        loadkw_new_all.append(loadkw_new)
        loadkvar_new = load_mult * loadkvar_dict[loadname]
        # now change opendss load object
        run_command('edit load.{ln} kw={kw}'.format(ln=loadname, kw=loadkw_new))
        run_command('edit load.{ln} kvar={kw}'.format(ln=loadname, kw=loadkvar_new))
        # sum+=loadkw_new
        dss.Loads.Next()

    print('@@@@@@@@@@Sum of Multiplied load is %s' %sum(loadkw_new_all))
    print(f"Total power base line is {dss.Circuit.TotalPower()}") # (read-only) Total power, watts delivered to the circuit

    P0, Q0 = dss.Circuit.TotalPower() 

    v0=
    a0=

    dss.Vsources.PU(v0) # Per-unit value of source voltage
    dss.Vsources.AngleDeg(np.degrees(a0)) # Phase angle of first phase in degrees

    i=1
    for ind in index_DERs:
        nodename = allnodenames[ind]
        # print(nodename)
        dss.Circuit.SetActiveBus(nodename)
        buskv = dss.Bus.kVBase()
        run_command(f'New generator.der_{i} bus1={nodename} Phases=1 Conn=Wye Model=1 kV={buskv} kw={DER_output[i-1]} kvar=0')
        i=i+1
    run_command("BatchEdit Generator..* Yearly=dummy")

    dss.run_command("Solve number=1 stepsize=1s")
    allbusmagpu = dss.Circuit.AllBusMagPu()
    allbusmagpu_base = np.array(allbusmagpu)

    print(dss.Circuit.TotalPower())
    

    return allbusmagpu_base
