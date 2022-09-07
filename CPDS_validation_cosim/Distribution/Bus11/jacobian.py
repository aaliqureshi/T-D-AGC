from time import time_ns
import numpy as np
import DSS_PF
import os
import pandas as pd
from set_DSS import dir_to_feeder


def calc (del_agc,typee,DER_idx,DER_node_idx,var_dict,num_nodes,P_diff):
    """performs calculations for sensitivity matrix. returns sensitivity matrix and DER sensitivity values"""
    dir_to_feeder = os.getcwd()
    # print(dir_to_feeder)
    dir_to_results = os.path.join(dir_to_feeder,'..','simulation_results')
    # print(dir_to_results)

    # voltage0_ref=pd.read_csv(dir_to_results+'\\initial_voltage_'+str(del_agc)+'.csv')

    # for idx in range (len(DER_idx)):
    #     var_dict[idx] = pd.read_csv(dir_to_results+'\\'+typee+'_results_'+str(del_agc)+'_'+str(idx)+'.csv')


    # jac_cols={}
    
    # start = time_ns()
    # for row in range(len(DER_idx)):
    #     arr=list()
    #     for col in range (num_nodes):
    #         # val = (var_dict[row].iloc[col,1] - voltage0_ref.iloc[4,col+1])/P_diff[row]
    #         val = (var_dict[row].iloc[col,1] - voltage0_ref.iloc[0,col+1])/P_diff[row]
    #         arr.append(val)
    #     jac_cols[row]=arr
    # end = time_ns()

    # t0 = (end-start)/1e6

    # print(f'Nested Loop executed in {t0} msec.')

    ###### just for fun testing ###################################

    ## using list comprehension!

    # arr=list()
    # jac_cols={}
    # start = time_ns()
    # i=0
    # for row in range(len(DER_idx)):
    #     val = [(var_dict[row].iloc[col,1] - voltage0_ref.iloc[0,col+1])/P_diff[row] for col in range(num_nodes)]
    #     jac_cols[i]=val[:]
    #     i+=1
    # end =time_ns()

    # t1=(end-start)/1e6

    # print(f'List Comprehension executed in {t1} millisec')


    # start = time_ns()
    # for row in range(len(DER_idx)):
    #     arr=list()
    #     for col in range (num_nodes):
    #         # val = (var_dict[row].iloc[col,1] - voltage0_ref.iloc[4,col+1])/P_diff[row]
    #         val = (var_dict[row].iloc[col,1] - voltage0_ref.iloc[0,col+1])/P_diff[row]
    #         arr.append(val)
    #     jac_cols[row]=arr
    # end = time_ns()

    jac_cols={}
    voltage0_ref=pd.read_csv(dir_to_results+'\\initial_voltage_'+str(del_agc)+'.csv')

    voltage0_np = voltage0_ref.iloc[0,1:].to_numpy()
    # var_np = np.array([])
    
    for idx in range (len(DER_idx)):
        val = pd.read_csv(dir_to_results+'\\'+typee+'_results_'+str(del_agc)+'_'+str(idx)+'.csv')
        var_dict[idx] = val.iloc[:,1].to_numpy()
    
    start=time_ns()
    i=0
    for col in range (len(DER_idx)):
        val = (var_dict[col] - voltage0_np)/P_diff[col]
        jac_cols[i]=val[:]
        i+=1
    end=time_ns()

    t2 = (end-start)/1e6

    print(f'--->> Jacobian computed in {t2} millisec.')



    ################################################################




    X = np.empty([len(DER_idx),len(jac_cols[0])])
    
    j=0
    for idx in range(len(DER_idx)):
        X[j]=jac_cols[idx]
        j+=1

    X=X.T # matrix transpose

    X_mat = np.asmatrix(X)

    X_results = pd.DataFrame(X)
    # dir_to_results = os.path.join(dir_to_feeder, "simulation_results")
    X_results.to_csv(dir_to_results+'\\'+typee+'_Sensitivity_'+str(del_agc)+'_'+typee+'.csv')



    col=0
    idx =0
    X_sens=list()
    while col< len(DER_idx):
        sens1 = X[DER_node_idx[idx],col]
        sens2 = X[DER_node_idx[idx+1],col]
        X_sens.append(min(sens1,sens2))
        col+=1
        idx+=2

    return X_mat,X_sens

    



def Tan(DER_idx,DER_node_idx,DER_pert,DER_output,del_agc):

    """function to build Jacobian matrix based on initial voltage and DER_pert (perturbed voltage)
       returns sensitivity matrix and sensitivity matrix values corresponding to DERs
       calc function performs the actual calculations
       solvePF solves power flow using OpenDSS and stores the results in a csv file used by calc function"""

    typee='tangent'
    var_dict ={}
    DER_out = DER_output[:]
    P_diff = [DER_pert[i] - DER_output[i] for i in range(len(DER_idx))]
    for i in range (len(DER_idx)):
        DER_out[i]=DER_pert[i]
        # print(DER_out)
        # num_nodes,_,_=DSS_PF.solvePF (DER_out,DER_idx,del_agc,type,i,store=1)
        num_nodes,_,_=DSS_PF.solvePF_8500_balanced (DER_out,DER_idx,del_agc,typee,i,store=1)
        DER_out[i]=DER_output[i]
        val='vt_d' + str(i)
        var_dict[i] = val
    T_mat,T_sens=calc(del_agc,typee,DER_idx,DER_node_idx,var_dict,num_nodes,P_diff)
    return T_mat,T_sens



def Sec(DER_idx,DER_node_idx,DER_max,DER_output,del_agc):
    """function to build Jacobian matrix based on initial voltage and DER_pert (perturbed voltage)
       returns sensitivity matrix and sensitivity matrix values corresponding to DERs
       calc function performs the actual calculations
       solvePF solves power flow using OpenDSS and stores the results in a csv file used by calc function"""

    typee='secant'
    var_dict={}
    DER_out = DER_output[:]
    P_diff=[DER_max[i]-DER_output[i] for i in range(len(DER_output))]
    for i in range (len(DER_idx)):
        DER_out[i]=DER_max[i]
        # print(DER_out)
        # num_nodes,_,_=DSS_PF.solvePF (DER_out,DER_idx,del_agc,type,i,store=1)
        num_nodes,_,_=DSS_PF.solvePF_8500_balanced (DER_out,DER_idx,del_agc,typee,i,store=1)
        DER_out[i]=DER_output[i]
        val='vs_d'+str(i)
        var_dict[i]= val
    S_mat,S_sens=calc(del_agc,typee,DER_idx,DER_node_idx,var_dict,num_nodes,P_diff)

    return S_mat,S_sens

