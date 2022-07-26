import numpy as np


def AGC_prop(DER_headroom,power_demand):
    """Function to compute proportional AGC allocation"""

    DER_headroom_prop = []
    # proportional factor
    alpha = power_demand/sum(DER_headroom)

    # check if power demand is greater than available headroom limit
    if alpha > 1:
        # find the difference between demand and availbile power
        val=power_demand - sum(DER_headroom)
        print("DERs can't meet the request. Maximum available power could be {}. \n Difference is {} ".format(sum(DER_headroom),val) )
        print('Setting requirement to %s' %sum(DER_headroom))
        alpha=1
    DER_headroom_prop=[alpha*val for val in DER_headroom]
    return DER_headroom_prop




def AGC_limit (V_max,DER_bus_voltage,DER_sens_list,num_DER):
    """compute effective headroom for each DER unit based on voltage profile"""
    DER_headroom_limit=list()
    for val in range(num_DER):
        DER_headroom_limit.append(((V_max - DER_bus_voltage[val]) / DER_sens_list[val] )/10)
    return DER_headroom_limit




def AGC_alloc (DER_headroom_limit, DER_headroom,num_DER):
    """find min. of AGC_prop & AGC_limit and set that value as power increase for each unit """
    del_pmat = [min(DER_headroom_limit[i],DER_headroom[i]) for i in range(num_DER)]
    return del_pmat



def solveLPF (del_pmat,Bus_voltage,num_bus,num_DER,X_mat,DER_idx):
    """function to calculate bus voltage based on sensitivity matrix and change in power (del_pmat)"""
    DER_bus_voltage_all =[]
    DER_bus_voltage=list()

    # perform M * del_pmat  
    for bus in range (num_bus):
        for col in range (num_DER):
            val = X_mat[bus,col] *  del_pmat[col]
        DER_bus_voltage_all.append(val)


    # print('size : %s' %len(DER_bus_voltage_all))

    Bus_voltage = [DER_bus_voltage_all[i]+Bus_voltage[i] for i in range(num_bus)]
    DER_voltage = [Bus_voltage[idx] for idx in DER_idx]

    # return only DER_bus voltages
    # j=0
    # for idx in num_bus:
    #     DER_bus_voltage_a.append(float(DER_bus_voltage_all[idx] + DER_bus_voltage_init[j]))
    #     j+=1

    return Bus_voltage, DER_voltage


def AGC_calculation (DER_headroom,del_power_demand,V_max,DER_sens_list,Bus_voltage,DER_idx,DER_output,X_mat):

    """main function to calculate AGC allocation based on headroom, max. power injection limit(voltage profile)
    returns bus voltage and DER output"""
    AGC_undelivered = True
    DER_bus_voltage =[Bus_voltage[idx] for idx in DER_idx]
    num_DER = len(DER_idx)
    num_bus = len(Bus_voltage)
    del_pmat=np.zeros(num_DER)
    del_power_demand_const=del_power_demand
    initial_output = sum(DER_output)

    while AGC_undelivered:

        print('DER headroom is %s' %DER_headroom)

        # find the proportional headroom for each DER unit
        DER_headroom_prop=AGC_prop(DER_headroom,del_power_demand)

        print(sum(DER_headroom_prop))

        print(DER_headroom_prop)

        # DER_bus_voltage = [voltage0_ref.iloc[10,5],voltage0_ref.iloc[10,11],voltage0_ref.iloc[10,13],voltage0_ref.iloc[10,16]]

        # find effective headroom for each DER unit
        DER_headroom_limit = AGC_limit(V_max,DER_bus_voltage,DER_sens_list,num_DER)
        print('DER headroom limit is %s'%DER_headroom_limit)

        # Allocate AGC to DER units based on AGC_prop & AGC_limit
        DER_headroom = AGC_alloc (DER_headroom_limit, DER_headroom,num_DER)

        print('DER headroom set as: %s' %DER_headroom)   

        # print('Sum of initial power allocation is %s' %sum(del_pmat))

        P_diff=0
        # allocate appropriate ower to each unit
        for i in range (num_DER):
            # check if eff < prop
            if DER_headroom[i] <= DER_headroom_prop[i]:

                print(DER_headroom_limit[i])

                print(DER_headroom_prop[i])
                # find power difference
                P_diff+=(DER_headroom_prop[i]-DER_headroom[i])
                # DER_sat_list.append(i)
                # increase the DER power output by DER_limit
                DER_output[i] = DER_output[i] + DER_headroom[i]
                # set DER headroom to 0!
                DER_headroom[i] = 0
                # set del_pmat as DER_limit
                del_pmat[i] = DER_headroom[i]
                # num_DER -=1
                # DER_headroom.pop[i]
            else:
                # increase DER power by DER_prop
                DER_output[i] = DER_output[i] + DER_headroom_prop[i]
                # set the available headroom as the diff. between available headroom & DER_prop 
                DER_headroom[i] = DER_headroom[i] - DER_headroom_prop[i]
                # set del_pmat as DER_prop
                del_pmat[i] = DER_headroom_prop[i]  
        
        print('~~~~DER output %s' %DER_output)

        print('DER headroom updated as: %s' %DER_headroom)
        del_power_demand = P_diff # update the power demand as the total undelivered power

        # perform linear power flow for node voltage 

        Bus_voltage,DER_bus_voltage = solveLPF(del_pmat, Bus_voltage, num_bus,num_DER,X_mat,DER_idx)

        # DER_bus_voltage = [DER_bus_voltage_const[i]+DER_bus_voltage[i] for i in range (num_DER)]

        print('bus voltages are %s' %DER_bus_voltage)

        if (P_diff == del_power_demand_const) :
            print("DER max. limit reached. \n Total power delivered is {}".format(sum(DER_output) - initial_output))
            AGC_undelivered = False
        elif P_diff == 0:
            AGC_undelivered = False
        else:
            continue
    
    return Bus_voltage,DER_output