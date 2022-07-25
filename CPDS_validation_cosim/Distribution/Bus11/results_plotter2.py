import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


total_buses=13


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


p1_j,p2_j,p3_j,p4_j,p5_j,p6_j= [],[],[],[],[],[]
p1_s,p2_s,p3_s,p4_s,p5_s,p6_s= [],[],[],[],[],[]

val=0
idx=1
for i in range (13):
    val= (abs(v1.iloc[4,idx+2] - s1.iloc[idx,1])/(v1.iloc[4,idx+2]))*100
    p1_s.append(val)
    val= (abs(v2.iloc[4,idx+2] - s2.iloc[idx,1])/(v2.iloc[4,idx+2]))*100
    p2_s.append(val)
    val= (abs(v3.iloc[4,idx+2] - s3.iloc[idx,1])/(v3.iloc[4,idx+2]))*100
    p3_s.append(val)
    val= (abs(v4.iloc[4,idx+2] - s4.iloc[idx,1])/(v4.iloc[4,idx+2]))*100
    p4_s.append(val)
    val= (abs(v5.iloc[4,idx+2] - s5.iloc[idx,1])/(v5.iloc[4,idx+2]))*100
    p5_s.append(val)
    val= (abs(v6.iloc[4,idx+2] - s6.iloc[idx,1])/(v6.iloc[4,idx+2]))*100
    p6_s.append(val)

    val= (abs(v1.iloc[4,idx+2] - j1.iloc[idx,1])/(v1.iloc[4,idx+2]))*100
    p1_j.append(val)
    val= (abs(v2.iloc[4,idx+2] - j2.iloc[idx,1])/(v2.iloc[4,idx+2]))*100
    p2_j.append(val)
    val= (abs(v3.iloc[4,idx+2] - j3.iloc[idx,1])/(v3.iloc[4,idx+2]))*100
    p3_j.append(val)
    val= (abs(v4.iloc[4,idx+2] - j4.iloc[idx,1])/(v4.iloc[4,idx+2]))*100
    p4_j.append(val)
    val= (abs(v5.iloc[4,idx+2] - j5.iloc[idx,1])/(v5.iloc[4,idx+2]))*100
    p5_j.append(val)
    val= (abs(v6.iloc[4,idx+2] - j6.iloc[idx,1])/(v6.iloc[4,idx+2]))*100
    p6_j.append(val)

    idx+=1

# error_j = np.array( [p1_j,p2_j,p3_j,p4_j,p5_j,p6_j])
# error_s = np.array([p1_s,p2_s,p3_s,p4_s,p5_s,p6_s])

p1_s_max=max(p1_s)
p2_s_max=max(p2_s)
p3_s_max=max(p3_s)
p4_s_max=max(p4_s)
p5_s_max=max(p5_s)
p6_s_max=max(p6_s)

p1_j_max=max(p1_j)
p2_j_max=max(p2_j)
p3_j_max=max(p3_j)
p4_j_max=max(p4_j)
p5_j_max=max(p5_j)
p6_j_max=max(p6_j)


p1_s_avg=sum(p1_s)/ len(p1_s)
p2_s_avg=sum(p2_s)/ len(p2_s)
p3_s_avg=sum(p3_s)/len(p3_s)
p4_s_avg=sum(p4_s)/len(p4_s)
p5_s_avg=sum(p5_s)/len(p5_s)
p6_s_avg=sum(p6_s)/len(p6_s)

p1_j_avg=sum(p1_j)/len(p1_j)
p2_j_avg=sum(p2_j)/len(p2_j)
p3_j_avg=sum(p3_j)/len(p3_j)
p4_j_avg=sum(p4_j)/len(p4_j)
p5_j_avg=sum(p5_j)/len(p5_j)
p6_j_avg=sum(p6_j)/len(p6_j)

j_max =np.array([p1_j_max,p2_j_max,p3_j_max,p4_j_max,p5_j_max,p6_j_max])
s_max =np.array([p1_s_max,p2_s_max,p3_s_max,p4_s_max,p5_s_max,p6_s_max])

j_avg =np.array([p1_j_avg,p2_j_avg,p3_j_avg,p4_j_avg,p5_j_avg,p6_j_avg])
s_avg =np.array([p1_s_avg,p2_s_avg,p3_s_avg,p4_s_avg,p5_s_avg,p6_s_avg])

print(j_max)

print(s_max)

print(j_avg)

print(s_avg)

x_axis=[1.05,1.2,1.6,2.0,2.4,2.8]
plt.figure(1)

plt.plot(x_axis,j_max, "--" , linewidth=3)
plt.plot(x_axis,s_max, "--", linewidth=3)
plt.plot(x_axis,j_avg,'-', linewidth=3)
plt.plot(x_axis,s_avg,'-', linewidth=3)
plt.xticks([1.05,1.2,1.6,2.0,2.4,2.8])
plt.ylim(top=0.06)
# plt.legend(('Max. error', 'Secant Line Approximation (max)', 'AVg (T)','Avg(S)'))
plt.legend(('Max. error (tangent line approx.)', 'Max. error (secant line approx.)',' Average error (tangent line approx)','Average error (secant line approx.)'))
plt.xlabel("DER power output [pu]",fontweight='bold')
plt.ylabel("Voltage error [%]",fontweight='bold')
plt.title("Mean voltage error as a function of DER output",fontweight='bold')
plt.savefig('RES3',dpi=300)

plt.figure(2)

plt.plot(x_axis,s_max, "--", linewidth=3)
# plt.plot(x_axis,s_max, "--")
plt.plot(x_axis,s_avg,'-', linewidth=3)
# plt.plot(x_axis,s_avg,'-')
plt.xticks([1.05,1.2,1.6,2.0,2.4,2.8])
plt.ylim(top=0.06)
# plt.legend(('Max. error', 'Secant Line Approximation (max)', 'AVg (T)','Avg(S)'))
plt.legend(('Max. error', ' Average error'))
plt.xlabel("DER power output [pu]",fontweight='bold')
plt.ylabel("Voltage error [%]",fontweight='bold')
plt.title("Mean voltage error using secant line approximation", fontweight='bold')
plt.savefig('RES2',dpi=300)

plt.show()