import opendssdirect as dss
from opendssdirect.utils import run_command
import numpy as np
import pandas as pd
import os

dir_to_feeder=os.getcwd()
V_low = 0.95
V_high = 1.05


def setDSS(num_DER, DER_output,del_agc):

    """function to set DSS environment and build the test case
       returns feeder head voltage, angle, bus idx with which DERs are connected, initial voltage at all nodes """

    run_command('clear')

    run_command ('compile IEEE8500/Master.dss')

    if dss.Text.Result() == '':
        print('------ Success for the test run -----')
    else:
        print('------ Opendss failed ----')
        print(f'Error is "{dss.Text.Result()}"!')
        return 0


    ## get bus names with which loads are attached

    num_loads = dss.Loads.Count()
    dss.Loads.First()
    loadbuses = []

    for idx in range(num_loads):
        loadbuses.append(dss.CktElement.BusNames())
        dss.Load.Next()
    

    # get load kw values
    # BASEKV = dss.Vsources.BasekV()
    # # loadname_all=[]
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
    # num_nodes = len(allnodenames)

    # load_mult = 1.0 # changeable
    # loadkw_new_all = []
    # dss.Loads.First()
    # # sum = 0

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

    # print('@@@@@@@@@@ Sum of Multiplied load is %s' %sum(loadkw_new_all))
    print(f"Total base line power (3-phase) before DER connection is {dss.Circuit.TotalPower()}") # (read-only) Total power, watts delivered to the circuit

    # P0, Q0 = dss.Circuit.TotalPower()

    # ss.PQ.alter(src='p0',idx='PQ_8',value=-P0/1e5)
    # ss.PQ.alter (src='q0',idx='PQ_8',value=-Q0/1e5)
    # ss.PFlow.run()
    # v0=ss.Bus.v.v[7]
    # a0=np.degrees(ss.Bus.a.v[7]) 

    # current_time = 0
    allbusmagpu_ts = []
    totalpower_ts = []

    consistent_random_object = np.random.RandomState(321)
    # index_DERs = consistent_random_object.randint(len(allnodenames), size=num_DER) # return random integers from low (inclusive) to high (exclusive)
    index_DERs = consistent_random_object.choice(len(loadbuses),size=num_DER)
    # index_DERs = [7,11,16,20]
    DER_buses = [allnodenames[idx] for idx in index_DERs]
    print(f'DERs are connected to {DER_buses} buses')


    i=1
    for ind in index_DERs:
        nodename = allnodenames[ind]
        # print(nodename)
        dss.Circuit.SetActiveBus(nodename)
        buskv = dss.Bus.kVBase()
        run_command(f'New generator.der_{i} bus1={nodename} Phases=1 Conn=Wye Model=1 kV={buskv} kw={DER_output[i-1]} kvar=0')
        i=i+1
    # run_command("BatchEdit Generator..* Yearly=dummy")

    # DER_output_pre = None
    # voltage = v0
    run_command('Solve')

    print(f'Network active power demand after DER connection is{dss.Circuit.TotalPower()}')
    # print(f"index_DERs = {index_DERs}")

    total_time=5

    for request_time in np.arange(0, total_time):
        # print("==============================")
        # print(f"current_time={request_time}")
        

        # der_output_array.append(DER_output[0])

        j=1
        for ind in index_DERs:
            nodename = allnodenames[ind]
            run_command(f'Edit generator.der_{j} kw={DER_output[j-1]}') # opendss positive sequence should defined as 1 phase
            j=j+1
        
        print(f"DER_output={DER_output} at time {request_time}")
        dss.run_command("Solve number=1 stepsize=1s")
        print(f'Network active power demand after DER connection is{dss.Circuit.TotalPower()}')

        allbusmagpu = dss.Circuit.AllBusMagPu()
        allbusmagpu_ts.append(allbusmagpu)
        totalpower_ts.append(dss.Circuit.TotalPower())

        # total power adjust amount of total DER_output
        S = dss.Circuit.TotalPower() # the total power is negative
        # S_net=dss.Circuit.TotalPower()

        # S[0] = S[0] - np.sum(DER_output) # adjust
        #
        # print(f"======Total power demand={dss.Circuit.TotalPower()}")
        # print(f"DER power is {np.sum(DER_output)}")
        # print(f"Overall after adjustment S={S}")
        initial_net_power = -S[0]
        # Q = -S[1]

        # P = P*scale_P
        # Q = Q*scale_Q

        # print('Net active power at time step %s is: %s' %(current_time,P))
        # print('Net reactive power comsumption at time step %s is: %s' %(current_time,Q))

        allbusmagpu_base = np.array(allbusmagpu)

        # for zz in range(len(allbusmagpu_base)):
        #     if allbusmagpu_base[zz] < V_low:
        #         print('Bus %s Voltage is lower than nominal'%allnodenames[zz])
        #     elif allbusmagpu_base[zz] > V_high:
        #         print("Bus %s Voltage is higher than nominal value" %allnodenames[zz])

        
        # if request_time==0:
        #     angle_deg=np.degrees(a0)
        #     voltage=v0
        # else:
        # ss.PQ.alter(src='p0',idx='PQ_8',value=-S_net[0]/1e5)
        # ss.PQ.alter(src='q0',idx='PQ_8',value=-S_net[1]/1e5)
            # pq_andes=ss.PQ.get('p0','PQ_8',attr='vin')
            # ss.setup()
        # ss.PFlow.run()
            # print('p0 is {}'.format(ss.PQ.get(src='p0',idx='PQ_8',attr='vin')))
            # print('Ppf is {}'.format(ss.PQ.get(src='Ppf', idx='PQ_8', attr='v')))
        # voltage=ss.Bus.v.v[7]
        # angle_deg=np.degrees(ss.Bus.a.v[7])
            # angle_deg=np.degrees(a_array[7])
            # voltage=v_array[7]
        
        # print('!!!!!PF voltage is %s' %voltage)
        # print('!!!!!PF angle is %s' %angle_deg)
        # dss.Vsources.AngleDeg(angle_deg)
        # dss.Vsources.PU(voltage)

        

    # now = datetime.now() # record the time
    # append_time = now.strftime("%m_%d_%Y_%H_%M_%S")

    voltage_results = pd.DataFrame(allbusmagpu_ts, columns=allnodenames)
    dir_to_results = os.path.join(dir_to_feeder, "simulation_results")
    # voltage_results.to_csv(dir_to_results+'\\voltage_results_'+str(active_der)+'_'+active_power+'_'+'.csv')
    voltage_results.to_csv(dir_to_results+'\\initial_voltage_'+str(del_agc)+'.csv')

    allnodenames_df = pd.DataFrame(allnodenames)
    allnodenames_df.to_csv(dir_to_results+'\\nodes_'+str(del_agc)+'.csv')


    
    # voltage = allbusmagpu_base[0]
    # angle_deg = 0

    # kk=voltage_results[allnodenames[0]]
    dss.Generators.First()
    for i in range (4):
        print('************************')
        print(f'DER {i} output is: {dss.Generators.kW()}')
        dss.Generators.Next()

    print("~~Init finalized")
    run_command('clear')
    return index_DERs,allbusmagpu_base,initial_net_power




def setDSS_8500_balanced(num_DER, DER_output,del_agc,feeder_type):

    run_command('compile IEEE8500/Master.dss')

        ### load buses selection based on bus list rather than load list ###

    loadbuses_index=list()
    allnodes = dss.Circuit.AllNodeNames()

    for idx in range(len(allnodes)):
        dss.Circuit.SetActiveBus(allnodes[idx])
        if len(dss.Bus.LoadList()) !=0:
            loadbuses_index.append(idx)

                            ### end ###

        ## get 2 phase bus name for a balanced 2-phase connection of DER

    num_loads = dss.Loads.Count()
    dss.Loads.First()
    loadbuses_two_phase = list()

    for idx in range(num_loads):
        # print(dss.Loads.Name())
        # print(dss.CktElement.BusNames())
        loadbuses_two_phase.append(dss.CktElement.BusNames())
        # print(dss.CktElement.NodeNames())
        dss.Loads.Next()

                                ## end ##

        ### generate random numbers to be set as DER indexes!

    seed = np.random.RandomState(321)
    # index = seed.randint(len(loadbuses),size = 100)

    DER_index_two_phase = seed.choice(len(loadbuses_two_phase),size=num_DER,replace=False) # replace is set as False so that no bus is selected twice

    DER_index =[]

    index_offset = loadbuses_index[0]-1

    for idx in range(len(DER_index_two_phase)):
        DER_index.append(index_offset + (DER_index_two_phase[idx]*2)+1)
        DER_index.append(index_offset + (DER_index_two_phase[idx]*2) +2)
    
    DER_node_idx = DER_index[:]

                                ## end


    print(f'Initial Power demand is {dss.Circuit.TotalPower()} kW.')

    print(f'{num_DER} DERs are to be connected to the system.')

    power0 = dss.Circuit.TotalPower()[0]

    # print(f'DER connected to {loadbuses[index2[0]][0]}')

        ### Initialize DERs in the system (DERs connected as 2-phase)
    j=0
    for idx in range (num_DER):
        run_command(f'New generator.der_{idx+1} bus1={loadbuses_two_phase[DER_index_two_phase[idx]][0]} phases =2 model =1 conn= wye kV=0.208 kW={DER_output[idx]} kvar = 0')
        # print(f'DER {idx+1} connected to bus {loadbuses_two_phase[DER_index_two_phase[idx]][0]}')
        # print(f'Corresponding nodes are: {allnodes[DER_index[j]]} and {allnodes[DER_index[j+1]]} ')
        j+=2

    run_command('solve')

    print(f'Power Demand after DER integration is {dss.Circuit.TotalPower()} kW.')

    power1= dss.Circuit.TotalPower()[0]

    print(f'Total power connected to the system is {sum(DER_output)} kW. Difference in power import is {-power0 + power1 } kW. ')


    allbusmagpu = dss.Circuit.AllBusMagPu()
    allbusmagpu_ts=[]
    allbusmagpu_ts.append(allbusmagpu)

    # S= dss.Circuit.TotalPower()

    initial_net_power=-power1

    voltage_results = pd.DataFrame(allbusmagpu_ts,columns=allnodes)
    dir_to_results = os.path.join(dir_to_feeder,"simulation_results")
    voltage_results.to_csv(dir_to_results+'\\initial_voltage_'+str(del_agc)+'.csv')

    allnodes_df = pd.DataFrame(allnodes)
    allnodes_df.to_csv(dir_to_results+'\\nodes_'+str(del_agc)+'.csv')

    run_command('clear')

    print('~~~~~~~~~~~~~Init Finalized~~~~~~~~~~~~~~')

    return DER_index_two_phase, DER_node_idx, allbusmagpu,initial_net_power
        









