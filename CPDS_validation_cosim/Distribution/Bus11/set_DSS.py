import opendssdirect as dss
from opendssdirect.utils import run_command
import andes
import numpy as np
import pandas as pd
import os

V_low = 0.95
V_high = 1.05


def setDSS(num_DER, DER_output,del_agc):

    dir_to_feeder = os.getcwd()
    path_to_tx_xlsx=os.path.join(dir_to_feeder,'..','..','Network_models','Transmission')

    ############ setting andes ####################
    tx_xlsx= 'ieee14_pvd1_1DistBu1.xlsx'
    tx_path=os.path.join(path_to_tx_xlsx,tx_xlsx)
    ss=andes.run(tx_path,setup=False)
    ss.setup()

    ########################################

    run_command ('compile DSSfiles/master.dss')
    if dss.Text.Result() == '':
        print('------ Success for the test run -----')
    else:
        print('------ Opendss failed ----')
        print(f'Error is "{dss.Text.Result()}"!')

    run_command("BatchEdit Load..* Vminpu=0.9")
    run_command("New Loadshape.dummy npts=60 sinterval=1 Pmult=[file=dummy_profile.txt]")
    run_command("BatchEdit Load..* Yearly=dummy")
    run_command("set mode=snapshot")
    dss.Solution.ControlMode(2)

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

    loadkw_dict = dict(zip(loadname_all, loadkw_all))
    loadkvar_dict = dict(zip(loadname_all, loadkvar_all))
    allnodenames = dss.Circuit.AllNodeNames()
    num_nodes = len(allnodenames)

    load_mult = 1.0111
    loadkw_new_all = []
    dss.Loads.First()
    # sum = 0

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

    # print('@@@@@@@@@@ Sum of Multiplied load is %s' %sum(loadkw_new_all))
    print(f"Total power base line is {dss.Circuit.TotalPower()}") # (read-only) Total power, watts delivered to the circuit

    P0, Q0 = dss.Circuit.TotalPower()

    ss.PQ.alter(src='p0',idx='PQ_8',value=-P0/1e5)
    ss.PQ.alter (src='q0',idx='PQ_8',value=-Q0/1e5)
    ss.PFlow.run()
    v0=ss.Bus.v.v[7]
    a0=np.degrees(ss.Bus.a.v[7]) 

    current_time = 0
    allbusmagpu_ts = []
    totalpower_ts = []

    consistent_random_object = np.random.RandomState(321)
    index_DERs = consistent_random_object.randint(len(allnodenames), size=num_DER) # return random integers from low (inclusive) to high (exclusive)


    # DER_output = np.ones(num_DER)*(DER_output_MW_prescale*1e3/num_DER)/scale_P

    # DER_output_limit=np.ones(num_DER)*(DER_rating_MW_prescale*1e3/num_DER)/scale_P

    # DER_output=np.zeros(num_DER)

    # print(f"DER output after scale in kw each = {DER_output[0]} of {num_DER} DERs.")

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

    DER_output_pre = None
    voltage = v0

    print(dss.Circuit.TotalPower())
    print(f"index_DERs = {index_DERs}")

    total_time=5

    for request_time in np.arange(0, total_time):
        print("==============================")
        print(f"current_time={request_time}")

        # if DER_output_pre is not None:# 
        #     if np.isscalar(DER_output_pre):
        #         DER_output = (DER_output_pre/num_DER)*np.ones(num_DER)
        #     else:
        #         # DER_output = (sum(DER_output_pre)/num_DER)*np.ones(num_DER)
        #         # DER_output = request_MW - sum(DER)
        #         DER_output = DER_output_pre
        #     # print(f'scale_P={scale_P}')
        print(f"DER_output={DER_output} at time {request_time}")

        # der_output_array.append(DER_output[0])

        j=1
        for ind in index_DERs:
            nodename = allnodenames[ind]
            run_command(f'Edit generator.der_{j} kw={DER_output[j-1]/3}') # opendss positive sequence should defined as 1 phase
            j=j+1

        dss.run_command("Solve number=1 stepsize=1s")

        allbusmagpu = dss.Circuit.AllBusMagPu()
        allbusmagpu_ts.append(allbusmagpu)
        totalpower_ts.append(dss.Circuit.TotalPower())

        # total power adjust amount of total DER_output
        S = dss.Circuit.TotalPower() # the total power is negative
        S_net=dss.Circuit.TotalPower()

        S[0] = S[0] - np.sum(DER_output) # adjust
        #
        print(f"======Total power demand={dss.Circuit.TotalPower()}")
        # print(f"DER power is {np.sum(DER_output)}")
        # print(f"Overall after adjustment S={S}")
        P = -S[0]
        Q = -S[1]

        # P = P*scale_P
        # Q = Q*scale_Q

        print('Net active power at time step %s is: %s' %(current_time,P))
        # print('Net reactive power comsumption at time step %s is: %s' %(current_time,Q))

        #Xeq = Q/voltage**2
        # now publish the results
        # h.helicsPublicationPublishComplex(pub, P, Q) # Xeq was Q
        # print("Sent Active power at time {}: {} kw".format(current_time, P))
        # print("Sent Reactive power at time {}: {} kvar".format(current_time, Q))

        allbusmagpu_base = np.array(allbusmagpu)
        # print(f'min of voltage = {min(allbusmagpu_base)}')
        # print(f'max of voltage = {max(allbusmagpu_base)}')

        # allbusmagpu_base = np.array(allbusmagpu)

        # print('Bus %s has the lowest voltage of %s'%(V_min,allnodenames[V_min_idx]))
        # print('Bus %s has the highest voltage of %s'%(V_max,allnodenames[V_max_idx]))

        for zz in range(len(allbusmagpu_base)):
            if allbusmagpu_base[zz] < V_low:
                print('Bus %s Voltage is lower than nominal'%allnodenames[zz])
            elif allbusmagpu_base[zz] > V_high:
                print("Bus %s Voltage is higher than nominal value" %allnodenames[zz])

        
        # if request_time==0:
        #     angle_deg=np.degrees(a0)
        #     voltage=v0
        # else:
        ss.PQ.alter(src='p0',idx='PQ_8',value=-S_net[0]/1e5)
        ss.PQ.alter(src='q0',idx='PQ_8',value=-S_net[1]/1e5)
            # pq_andes=ss.PQ.get('p0','PQ_8',attr='vin')
            # ss.setup()
        ss.PFlow.run()
            # print('p0 is {}'.format(ss.PQ.get(src='p0',idx='PQ_8',attr='vin')))
            # print('Ppf is {}'.format(ss.PQ.get(src='Ppf', idx='PQ_8', attr='v')))
        voltage=ss.Bus.v.v[7]
        angle_deg=np.degrees(ss.Bus.a.v[7])
            # angle_deg=np.degrees(a_array[7])
            # voltage=v_array[7]
        
        print('!!!!!PF voltage is %s' %voltage)
        print('!!!!!PF angle is %s' %angle_deg)
        dss.Vsources.AngleDeg(angle_deg)
        dss.Vsources.PU(voltage)

        

    # now = datetime.now() # record the time
    # append_time = now.strftime("%m_%d_%Y_%H_%M_%S")

    voltage_results = pd.DataFrame(allbusmagpu_ts, columns=allnodenames)
    dir_to_results = os.path.join(dir_to_feeder, "simulation_results")
    # voltage_results.to_csv(dir_to_results+'\\voltage_results_'+str(active_der)+'_'+active_power+'_'+'.csv')
    voltage_results.to_csv(dir_to_results+'\\initial_voltage_'+str(del_agc)+'.csv')

    print("Init finalized")

    # kk=voltage_results[allnodenames[0]]

    return voltage,angle_deg,index_DERs,allbusmagpu_base


