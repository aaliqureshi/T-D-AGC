import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## plot results
v1=pd.read_csv('voltage_results_525.csv')
v2=pd.read_csv('voltage_results_600.csv')
v3=pd.read_csv('voltage_results_800.csv')
v4=pd.read_csv('voltage_results_1000.csv')
v5=pd.read_csv('voltage_results_1200.csv')
v6=pd.read_csv('voltage_results_1400.csv')

j1=pd.read_csv('jacobian_results_525.csv')
j2=pd.read_csv('jacobian_results_600.csv')
j3=pd.read_csv('jacobian_results_800.csv')
j4=pd.read_csv('jacobian_results_1000.csv')
j5=pd.read_csv('jacobian_results_1200.csv')
j6=pd.read_csv('jacobian_results_1400.csv')

s1=pd.read_csv('secant_results_525.csv')
s2=pd.read_csv('secant_results_600.csv')
s3=pd.read_csv('secant_results_800.csv')
s4=pd.read_csv('secant_results_1000.csv')
s5=pd.read_csv('secant_results_1200.csv')
s6=pd.read_csv('secant_results_1400.csv')
# print(r1.iloc(3,3))

plt.figure (1)

# for i in range (15)
j=3
x=[632,670,671,680,633,645,646,692,675,684,611,652,634]
agc_request=[100,200,400,900,1050]
# y=[r1.iloc(4,3),r1.iloc(4,3),r1.iloc(4,3)]

df1,df2,df3,df4,df5= [],[],[],[],[]

val=0
idx=1
for i in range (13):
    val= abs(r1.iloc[4,idx+2] - s1.iloc[idx,1])
    df1.append(val)
    val= abs(r2.iloc[4,idx+2] - s2.iloc[idx,1])
    df2.append(val)
    val= abs(r3.iloc[4,idx+2] - s3.iloc[idx,1])
    df3.append(val)
    val= abs(r4.iloc[4,idx+2] - s4.iloc[idx,1])
    df4.append(val)
    val= abs(r5.iloc[4,idx+2] - s5.iloc[idx,1])
    df5.append(val)
    idx+=1

# for xx in range (len(x)):
#     plt.plot(x[xx],(r1.iloc[4,j]))
#     j+=1

error_np = np.array( [df1,df2,df3,df4,df5])

# error_ts = np.array([])
# for j in range (len(df1)):
#     for i in range (len(x)):
#         arr.append(df1[i])



print(error_np.shape)


# plt.figure(1)
# plt.stem(x,df1)
# plt.title('Difference between actual & computed results (P=100)')
# plt.xlabel('Bus Number')
# plt.ylabel('Difference in Voltage ')

# plt.figure(2)
# plt.stem(x,df2)
# plt.title('Difference between actual & computed results (P=200)')
# plt.xlabel('Bus Number')
# plt.ylabel('Difference in Voltage ')

# plt.figure(3)
# plt.stem(x,df3)
# plt.title('Difference between actual & computed results (P=400)')
# plt.xlabel('Bus Number')
# plt.ylabel('Difference in Voltage ')
# plt.figure(4)
# plt.stem(x,df4)
# plt.title('Difference between actual & computed results (P=900)')
# plt.xlabel('Bus Number')
# plt.ylabel('Difference in Voltage ')
# plt.figure(5)
# plt.stem(x,df5)
# plt.title('Difference between actual & computed results (P=1050)')
# plt.xlabel('Bus Number')
# plt.ylabel('Difference in Voltage ')


Bus=634
plt.figure(1)
plt.stem(agc_request,error_np[:,12])
plt.xticks([100,200,400,900,1050])
# plt.ylim(min(agc_request,error_np[:,1]),max(agc_request,error_np[:,1]))
plt.title('Error in Voltage Calculation at Bus %s' %Bus)
plt.xlabel('AGC request (kW)')
plt.ylabel('Voltage Error (pu)')
fig_name='Bus %s error'%Bus

plt.savefig(fig_name,dpi=300)


# plt.figure(6)
# # markerline1, stemlines, _ = plt.stem(x1, s_n_hat[0:5], '-.')
# markerline1, stemlines, _ =plt.stem(x[0],df1[0],'-.')
# plt.setp(markerline1, 'markerfacecolor', 'b')
# markerline2, stemlines, _ =plt.stem(x[0],df2[0],'-.')
# plt.setp(markerline2, 'markerfacecolor', 'r')
# markerline3, stemlines, _ =plt.stem(x[0],df3[0],'-.')
# plt.setp(markerline3, 'markerfacecolor', 'c')
# markerline4, stemlines, _ =plt.stem(x[0],df4[0],'-.')
# plt.setp(markerline4, 'markerfacecolor', 'm')
# markerline5, stemlines, _ =plt.stem(x[0],df5[0],'-.')
# plt.setp(markerline5, 'markerfacecolor', 'y')
# plt.stem(x[0],df2[0])
# plt.stem(x[0],df3[0])
# plt.stem(x[0],df4[0])
# plt.stem(x[0],df5[0])

# plt.show()