import opendssdirect as dss
from opendssdirect.utils import run_command
import numpy as np


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

DER_index_two_phase = seed.choice(len(loadbuses_two_phase),size=100,replace=False) # replace is set as False so that no two buses are selected twice

DER_index =[]

index_offset = loadbuses_index[0]-1

for idx in range(len(DER_index_two_phase)):
    DER_index.append(index_offset + (DER_index_two_phase[idx]*2)+1)
    DER_index.append(index_offset + (DER_index_two_phase[idx]*2) +2)

                            ## end


print(dss.Circuit.TotalPower())
# print(f'DER connected to {loadbuses[index2[0]][0]}')

     ### Initialize DERs in the system (DERs connected as 2-phase)
j=0
for idx in range (100):
    run_command(f'New generator.der_{idx+1} bus1={loadbuses_two_phase[DER_index_two_phase[idx]][0]} phases =2 model =1 conn= wye kV=0.208 kW=3 kvar = 0')
    # print(f'DER {idx+1} connected to bus {loadbuses_two_phase[DER_index_two_phase[idx]][0]}')
    # print(f'Corresponding nodes are: {allnodes[DER_index[j]]} and {allnodes[DER_index[j+1]]} ')
    j+=2

run_command('solve')

print(f'Power Demand after DER integration is {dss.Circuit.TotalPower()}')



