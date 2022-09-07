import numpy as np
from time import time_ns
from pulp import *


def AGC_prop(DER_headroom,power_demand):
    """Function to compute proportional AGC allocation"""

    DER_headroom_prop = []
    # proportional factor
    alpha = power_demand/sum(DER_headroom)
    power_diff=0

    # if alpha < 0.001:
    #     DER_max_headroom_idx = np.argmax(DER_headroom)
    #     if DER_headroom[DER_max_headroom_idx] > power_demand:
    #         DER_headroom[DER_max_headroom_idx] = alpha*DER_headroom[DER_max_headroom_idx]
    #     return DER_headroom_prop, power_diff 

    # check if power demand is greater than available headroom limit
    if alpha > 1:
        # find the difference between demand and availbile power
        power_diff=power_demand - sum(DER_headroom)
        print("^^^^^^^^^^^^^^^^^^^DERs can't meet the request. Maximum available power could be {}. \n Difference is {} ".format(sum(DER_headroom),power_diff) )
        # print('^^^^^^^^^^^^^^^^^^^Setting requirement to %s' %sum(DER_headroom))
        alpha=1
    DER_headroom_prop=[alpha*val for val in DER_headroom]
    return DER_headroom_prop,power_diff




def AGC_limit (V_max,DER_bus_voltage,DER_sens_list,num_DER):
    """compute effective headroom for each DER unit based on voltage profile"""
    DER_headroom_limit=list()
    for val in range(num_DER):
        DER_headroom_limit.append(((V_max - DER_bus_voltage[val]) / DER_sens_list[val] )/10)
    return DER_headroom_limit


def AGC_limit_max (V_max,Bus_voltage,X_mat,num_DER):
    """compute effective headroom based on jacobian inverse"""
    DER_headroom_limit=np.array([])
    V_max_array = np.ones(len(Bus_voltage)) * V_max
    delta_V = np.subtract(V_max_array,Bus_voltage)

    DER_headroom_limit = np.linalg.pinv(X_mat) @ delta_V

    # DER_headroom_limit = DER_headroom_limit 

    DER_headroom_limit = np.asarray(DER_headroom_limit).reshape(-1)

    return DER_headroom_limit





def AGC_alloc (DER_headroom_limit, DER_headroom,num_DER):
    """find min. of AGC_prop & AGC_limit and set that value as power increase for each unit """
    del_pmat = [min(DER_headroom_limit[i],DER_headroom[i]) for i in range(num_DER)]
    return del_pmat



def solveLPF (del_pmat,Bus_voltage,num_bus,num_DER,X_mat,DER_node_idx):
    """function to calculate bus voltage based on sensitivity matrix and change in power (del_pmat)"""
    DER_bus_voltage_all =[]
    # DER_bus_voltage=list()

    # perform M * del_pmat 
    # calculate time taken to solve LPF
    # start = time_ns()

    # for bus in range (num_bus):
    #     val=0
    #     for col in range (num_DER):
    #         val = val + X_mat[bus,col] *  del_pmat[col]
    #     DER_bus_voltage_all.append(val)

    DER_bus_voltage_all = X_mat @  del_pmat # DER_bus_voltage_all.shape = (1,8531)

    # DER_bus_voltage_all_2 = np.reshape(DER_bus_voltage_all_1, num_bus)



    # end = time_ns()

    # t = (end - start)/1e6

    # print(f'LPF solved in {t} millisec.')

    # print('size : %s' %len(DER_bus_voltage_all))

    Bus_volt = [DER_bus_voltage_all[0,i]+Bus_voltage[i] for i in range(num_bus)]
    # DER_voltage = [Bus_voltage[idx] for idx in DER_node_idx]

    # return only DER_bus voltages
    # j=0
    # for idx in num_bus:
    #     DER_bus_voltage_a.append(float(DER_bus_voltage_all[idx] + DER_bus_voltage_init[j]))
    #     j+=1

    return Bus_volt



def AGC_limit_LP (V_max,Bus_voltage,X_mat,num_DER,DER_headroom,DER_output,AGC_request):
    
    opt_model = LpProblem('get_DER_limit',LpMaximize)
    num_nodes = len(Bus_voltage)
    set_x = range(0,num_DER)
    set_v = range(0,num_nodes)
    v_l_ANSI = 0.95
    v_u_ANSI = V_max

    v_l = {i: v_l_ANSI - Bus_voltage[i] for i in set_v}
    v_u = {i:v_u_ANSI - Bus_voltage[i] for i in set_v}


    x_vars = {i: LpVariable(cat = LpContinuous, lowBound=0, upBound=DER_headroom[i],\
              name="x_{}".format(i)) for i in set_x}
    
    constraints = {i: opt_model.addConstraint(LpConstraint(\
                e = lpSum([X_mat[i,j]*x_vars[j] for j in set_x]),\
                sense=LpConstraintLE,\
                rhs = v_u[i],\
                name = 'constraint_v_u{}'.format(i)))\
                for i in set_v}

    constraints = {0: opt_model.addConstraint(LpConstraint(\
            e = lpSum([x_vars[j] for j in set_x]),\
            sense=LpConstraintEQ,\
            rhs = AGC_request,\
            name = 'constraint_p{}'.format(0)))\
            }

    # constraints = {i: opt_model.addConstraint(LpConstraint(\
    #             e = lpSum(X_mat[i,j]*x_vars[j] for j in set_x),\
    #             sense=LpConstraintGE,\
    #             rhs = v_l[i],\
    #             name = 'constraint_v_l{}'.format(i)))\
    #             for i in set_v}
    
    objective = lpSum([x_vars[i] for i in set_x])

    opt_model.setObjective(objective)

    t_start = time_ns()

    opt_model.solve()

    t_end = time_ns()

    print(f'--->LP solved in {(t_end - t_start)/1e6} milli-sec')

    # print('=======> Optimization Status:%s'%LpStatus[opt_model.status])

    # print('Sum of total injectable power is: ',value(opt_model.objective))

    limit =[]

    for v in opt_model.variables():
        print(v.name, '=',v.varValue)
        limit.append(v.varValue)
    
    return limit



def AGC_limit_LP_mult (V_max,Bus_voltage,X_mat,num_DER,DER_headroom,DER_output,AGC_request):

    V_min = 0.95
    # V_max = 1.05

    prob = LpProblem('OPF-type-AGC_distribution',LpMaximize)

    volt_limit = {i:V_max-Bus_voltage[i] for i in range(len(Bus_voltage))}

    xx=[V_max-Bus_voltage[i] for i in range(len(Bus_voltage))]


    x_vars = {i: LpVariable(cat = LpContinuous, lowBound=0, upBound=DER_headroom[i],\
              name = 'x_{}'.format(i)) for i in range(num_DER)}
    
    
    prob +=(lpSum([x_vars[i] for i in range(num_DER)]),"Sum of DER power output ",)

    for row in range (len(Bus_voltage)):
        prob +=(lpSum([X_mat[row,col]*x_vars[col] for col in range (num_DER)]) <= xx[row], "Voltage limit %s,%s"%(row,row*2),)
    
    prob += (lpSum([x_vars[i] for i in range(num_DER)]) == AGC_request)

    prob.solve()
    print('=======> Optimization Status:%s'%LpStatus[prob.status])

    # print('Sum of total injectable power is: ',value(prob.objective))

    limit = []

    for v in prob.variables():
        # print(v.name, '=',v.varValue)
        limit.append(v.varValue)
    
    return limit

def AGC_limit_LP_mult_cost (V_max,Bus_voltage,X_mat,num_DER,DER_headroom,DER_output,AGC_request):

    V_min = 0.95
    # V_max = 1.05

                    ## Calculate Cost Functions ## 

    # [1] https://www.eia.gov/outlooks/aeo/assumptions/pdf/table_8.2.pdf
    # [2] Economic dispatch for a microgrid considering renewable energy cost functions

    interest_rate =  0.09
    investment_life = 20
    investment_cost = 1748 # $/kW
    om_cost = 0.0038 # $/Kwh

    investment_coeff = interest_rate/abs(1-(1+interest_rate)**-investment_life)

    var1 = investment_coeff*investment_cost

    var2 = om_cost



    prob = LpProblem('Cost_Based_AGC_distribution',LpMinimize)

    volt_limit = {i:V_max-Bus_voltage[i] for i in range(len(Bus_voltage))}

    xx=[V_max-Bus_voltage[i] for i in range(len(Bus_voltage))]


    x_vars = {i: LpVariable(cat = LpContinuous, lowBound=0, upBound=DER_headroom[i],\
              name = 'x_{}'.format(i)) for i in range(num_DER)}
    
    
    prob +=(lpSum([var1*x_vars[i] + om_cost*x_vars[i] for i in range(num_DER)]),"Sum of DER power output cost ",)

    for row in range (len(Bus_voltage)):
        prob +=(lpSum([X_mat[row,col]*x_vars[col] for col in range (num_DER)]) <= xx[row], "Voltage limit %s,%s"%(row,row*2),)
    
    prob += (lpSum([x_vars[i] for i in range(num_DER)]) == AGC_request)

    prob.solve()
    print('=======> Optimization Status:%s'%LpStatus[prob.status])

    # print('Sum of total injectable power is: ',value(prob.objective))

    limit = []

    for v in prob.variables():
        # print(v.name, '=',v.varValue)
        limit.append(v.varValue)
    
    return limit




def Optimize (DER_headroom,AGC_request,V_max,Bus_voltage,DER_output,X_mat,DER_node_idx):

    num_DER = len(DER_headroom)
    num_bus = len(Bus_voltage)
    Bus_volt = Bus_voltage[:]
    DER_head = DER_headroom[:]

                                ## OPtimization Based Approach

    DER_headroom_limit_LP1=AGC_limit_LP (V_max,Bus_volt,X_mat,num_DER,DER_head,DER_output,AGC_request)
    DER_headroom_limit_LP2=AGC_limit_LP_mult_cost (V_max,Bus_volt,X_mat,num_DER,DER_head,DER_output,AGC_request)

    print('Optimization [W] results ---> Power increase is : %s'%sum(DER_headroom_limit_LP1))
    print('Optimization [A] results ---> Power increase is : %s'%sum(DER_headroom_limit_LP2))

    jj = sum([int(DER_headroom_limit_LP1[i]!=int(DER_headroom_limit_LP2[i])) for i in range(len(DER_headroom_limit_LP1))])

    print('Difference between W & A is %s' %jj)


    Bus_volt_W= solveLPF(DER_headroom_limit_LP1, Bus_volt, num_bus,num_DER,X_mat,DER_node_idx)
    Bus_volt_A = solveLPF(DER_headroom_limit_LP2, Bus_volt, num_bus,num_DER,X_mat,DER_node_idx)

    gp_w = sum([Bus_volt_W[i]>V_max for i in range(len(Bus_voltage))])
    gp_a = sum([Bus_volt_A[i]>V_max for i in range(len(Bus_voltage))])

    print('Voltage Violations in Optimization [W] are %s' %gp_w)
    print('Voltage Violations in Optimization [A] are %s' %gp_a)

    DER_output_LP = [DER_output[i]+DER_headroom_limit_LP2[i] for i in range(num_DER)]

    return DER_output_LP







def AGC_calculation (DER_headroom,del_power_demand,V_max,DER_sens_list,Bus_voltage,DER_idx,DER_node_idx,DER_output,X_mat):

    """main function to calculate AGC allocation based on headroom, max. power injection limit(voltage profile)
    returns bus voltage and DER output"""

    ## a copy is made to avoid mutation of original list
    DER_out=DER_output[:]
    DER_head=DER_headroom[:]
    DER_head_LP=DER_head[:]
    Bus_volt=Bus_voltage[:]
    ##

    AGC_undelivered = True
    DER_bus_voltage =[Bus_voltage[idx] for idx in DER_node_idx]
    num_DER = len(DER_headroom)
    num_bus = len(Bus_voltage)
    del_pmat=np.zeros(num_DER)
    del_power_demand_const=del_power_demand
    initial_output = sum(DER_output)
    ii=0 # iterator to check max_iter
    max_iter = 100
    solution_time=list()

    DER_out_LP=Optimize(DER_headroom,del_power_demand,V_max,Bus_voltage,DER_output,X_mat,DER_node_idx)

                            ## OPtimization Based Approach

    # DER_headroom_limit_LP1=AGC_limit_LP (V_max,Bus_voltage,X_mat,num_DER,DER_headroom,DER_output,del_power_demand_const)
    # DER_headroom_limit_LP2=AGC_limit_LP_mult (V_max,Bus_voltage,X_mat,num_DER,DER_headroom,DER_output,DER_sens_list,del_power_demand_const)

    # print('Optimization [W] results ---> Power increase is : %s'%sum(DER_headroom_limit_LP1))
    # print('Optimization [A] results ---> Power increase is : %s'%sum(DER_headroom_limit_LP2))

    # jj = sum([int(DER_headroom_limit_LP1[i]!=int(DER_headroom_limit_LP2[i])) for i in range(len(DER_headroom_limit_LP1))])

    # print('Difference between W & A is %s' %jj)

    # Bus_volt,DER_bus_voltage,t = solveLPF(del_pmat, Bus_volt, num_bus,num_DER,X_mat,DER_node_idx)

                                    ## end  ##########

    while AGC_undelivered:

        # print('DER headroom is %s' %DER_head)

        # find the proportional headroom for each DER unit
        t_start = time_ns()
        DER_headroom_prop,power_diff_prop=AGC_prop(DER_head,del_power_demand)

        # print(sum(DER_headroom_prop))

        # print(f'Proportional AGC allocation :{DER_headroom_prop}')


        # DER_bus_voltage = [voltage0_ref.iloc[10,5],voltage0_ref.iloc[10,11],voltage0_ref.iloc[10,13],voltage0_ref.iloc[10,16]]

        # find effective headroom for each DER unit

        # DER_headroom_limit = AGC_limit(V_max,DER_bus_voltage,DER_sens_list,num_DER)

        # gp = sum([Bus_volt[i]>V_max for i in range(len(Bus_voltage))])

        # for i in range(len(Bus_voltage)):
        #     if Bus_volt[i] > V_max:
        #         DER_head_LP[i] = 0

        DER_headroom_limit_max = AGC_limit_max (V_max,Bus_volt,X_mat,num_DER)

        # print('Proposed results are %s' %DER_headroom_limit_max)

        

        # print('$$$$$$$$$ ####### ######## %s &&&&&&&&&&&&&&&&&&&&& *****************' %gp)

        # print('DER headroom limit : %s'%DER_headroom_limit)

        # Allocate AGC to DER units based on AGC_prop & AGC_limit
        DER_head = AGC_alloc (DER_headroom_limit_max, DER_head,num_DER)

        # print('DER headroom updated to: %s' %DER_head)   

        # print('Sum of initial power allocation is %s' %sum(del_pmat))

        P_diff=power_diff_prop
        # allocate appropriate ower to each unit
        for i in range (num_DER):
            # check if eff < prop
            if DER_head[i] <= DER_headroom_prop[i]:

                # print(DER_headroom_limit[i])

                # print(DER_headroom_prop[i])
                # find power difference
                P_diff+=(DER_headroom_prop[i]-DER_head[i])
                # DER_sat_list.append(i)
                # increase the DER power output by DER_limit
                DER_out[i] = DER_out[i] + DER_head[i]
                # set del_pmat as DER_limit
                del_pmat[i] = DER_head[i]
                # set DER headroom to 0!
                DER_head[i] = 0
                # num_DER -=1
                # DER_headroom.pop[i]
            else:
                # increase DER power by DER_prop
                DER_out[i] = DER_out[i] + DER_headroom_prop[i]
                # set the available headroom as the diff. between available headroom & DER_prop 
                DER_head[i] = DER_head[i] - DER_headroom_prop[i]
                # set del_pmat as DER_prop
                del_pmat[i] = DER_headroom_prop[i]  
        
        # print('DER output updated to:  %s' %DER_out)

        # print('DER headroom updated to: %s' %DER_head)
        del_power_demand = P_diff # update the power demand as the total undelivered power



        # perform linear power flow for node voltage 

        # time_start =time_ns() 
        # print(time_start)
        Bus_volt = solveLPF(del_pmat, Bus_volt, num_bus,num_DER,X_mat,DER_node_idx)
        # solution_time.append(t)
        # time_end=time_ns()
        # print(time_end)
        # elap = time_end - time_start
        # print(elap)

        # print(f'LPF {ii+1} solved in {elap} sec.')
        # DER_bus_voltage = [DER_bus_voltage_const[i]+DER_bus_voltage[i] for i in range (num_DER)]

        # print('bus voltages are %s' %DER_bus_voltage)

        if (P_diff == del_power_demand_const) :
            # print("DER max. limit reached. \n Total power delivered is {}".format(sum(DER_out) - initial_output))
            print('DERs are not availble at this time.')
            AGC_undelivered = False
        elif sum(DER_output)+del_power_demand_const == sum(DER_out) or P_diff <= 1:
            t_end = time_ns()
            print(f'--->> Proposed Method solved in {(t_end - t_start)/1e6 } milli-sec')
            print('_+_+_+_+_+_Requested power dispatched!_+_+_+_+_+_')
            print(f'*_*_* Total Iterations: {ii+1} *_*_*')
            gp = sum([Bus_volt[i]>V_max for i in range(len(Bus_voltage))])
            print('Voltage violations in proposed method are %s' %gp)
            AGC_undelivered = False
        elif sum(DER_head) == 0:
            print(f'********DER max. limit reached. Total power dispatched is {sum(DER_out)-sum(DER_output)} kW. Total Iterations are {ii+1} ********')
            AGC_undelivered = False
        elif ii > max_iter:
            print('!!!!!!!!!Max. iterations reached!!!!!')
            AGC_undelivered = False
        else:
            ii+=1
            continue
    
    # return Bus_volt,DER_out,sum(solution_time)/len(solution_time),DER_out_LP
    return Bus_volt,DER_out,DER_out_LP