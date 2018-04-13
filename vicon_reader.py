# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:21:11 2018

@author: gabych
"""
import c3d
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from matplotlib.lines import Line2D   

# reading data from vicon
reader = c3d.Reader(open('/home/gabych/Documents/ETH/gaitIdentification/Vicon/Kai/Level.c3d','rb'))
points = []  # 13 markers, 5 values (x,y,z, residual value(estimate of the error for this particular point), cameras value(number of cameras that observed the point))
vgrf = [] # vertical ground reaction force
for i, point, analog in reader.read_frames():
    points.append(point)
    vgrf.append(analog)

# saving markers coordinates for visualiztion per frame  
#[L_TRC, R_TRC, COM, R_KNE, R_UIM, L_KNE, L_UIM, L_ANK, L_LIM, R_ANK, R_LIM, R_MT5, L_MT5]  
x = np.zeros((87500,13,1))
y = np.zeros((87500,13,1))
z = np.zeros((87500,13,1))
for p in range(0,len(points)):
    for j in range(0,len(points[0])):
        x[p][j][:]=points[p][j][0]
        y[p][j][:]=points[p][j][1]
        z[p][j][:]=points[p][j][2]
        
# to save the analog data in the corresponding value (Forces and moments per coordinate and per force plate)
Fx1 = []
Fy1 = []
Fz1 = []
Mx1 = []
My1 = []
Mz1 = []
Fx2 = []
Fy2 = []
Fz2 = []
Mx2 = []
My2 = []
Mz2 = []
for f in range (0,len(vgrf)):
    Fx1.append(vgrf[f][0])
    Fy1.append(vgrf[f][1])
    Fz1.append(vgrf[f][2][2])
    Mx1.append(vgrf[f][3])
    My1.append(vgrf[f][4])
    Mz1.append(vgrf[f][5])
    Fx2.append(vgrf[f][6])
    Fy2.append(vgrf[f][7])
    Fz2.append(vgrf[f][8][0])
    Mx2.append(vgrf[f][9])
    My2.append(vgrf[f][10])
    Mz2.append(vgrf[f][11])
    
# saving markers coordinates per element for joint angle calculation
L_TRC = []
R_TRC = []
C_TRC = []
COM = []
R_KNE = []
R_UIM = []
L_KNE = []
L_UIM = []
L_ANK = []
L_LIM = []
R_ANK = []
R_LIM = []
R_MT5 = []
L_MT5 = []

for p in range(0,len(points)):
    L_TRC.append(points[p][0][0:3])
    R_TRC.append(points[p][1][0:3])
    C_TRC.append((points[p][1][0:3]+points[p][0][0:3])/2)
    COM.append(points[p][2][0:3])
    R_KNE.append(points[p][3][0:3])
    R_UIM.append(points[p][4][0:3])
    L_KNE.append(points[p][5][0:3])
    L_UIM.append(points[p][6][0:3])
    L_ANK.append(points[p][7][0:3])
    L_LIM.append(points[p][8][0:3])
    R_ANK.append(points[p][9][0:3])
    R_LIM.append(points[p][10][0:3])
    R_MT5.append(points[p][11][0:3])
    L_MT5.append(points[p][12][0:3])
        

# joint angle calculation
a_r_knee_angle = []
a_l_knee_angle = []
a_r_knee_angle2 = []
a_l_knee_angle2 = []
b_r_knee_angle = []
b_l_knee_angle = []
r_hip_angle = []
l_hip_angle = []
trunk_angle = []
l_shank_angle = []
r_shank_angle = []
l_thigh_angle = []
r_thigh_angle = []
l_thigh_angle1 = []
r_thigh_angle1 = []

# angle calculation in sagital plane
for l in range(0,len(R_KNE)):
    l_shank_angle.append((np.arctan2((L_KNE[l][2]-L_ANK[l][2]), (L_KNE[l][1]-L_ANK[l][1])))*180/np.pi)
    r_shank_angle.append((np.arctan2((R_KNE[l][2]-R_ANK[l][2]), (R_KNE[l][1]-R_ANK[l][1])))*180/np.pi)
    l_thigh_angle.append(180-((np.arctan2((L_TRC[l][2]-L_KNE[l][2]), (L_TRC[l][1]-L_KNE[l][1])))*180/np.pi))
    r_thigh_angle.append(180-((np.arctan2((R_TRC[l][2]-R_KNE[l][2]), (R_TRC[l][1]-R_KNE[l][1])))*180/np.pi))
    trunk_angle.append(math.degrees(math.atan((C_TRC[l][2]-COM[l][2]/C_TRC[l][1]-COM[l][1]))))
    a_r_knee_angle.append(r_thigh_angle[l]-r_shank_angle[l])
    a_l_knee_angle.append(l_thigh_angle[l]-l_shank_angle[l])
    b_r_knee_angle.append(r_shank_angle[l]+(180-r_thigh_angle[l]))
    b_l_knee_angle.append(l_shank_angle[l]+(180-l_thigh_angle[l]))
    r_hip_angle.append(r_thigh_angle[l]-trunk_angle[l])
    l_hip_angle.append(l_thigh_angle[l]-trunk_angle[l])
    
# filter grf data
Fz_r= np.asarray(Fz1[:18000])
Fz_r= Fz_r[Fz_r >= 0]
step_valleys = np.asarray(np.where(Fz_r==0))
step_points_init = step_valleys[:,np.where(np.diff(step_valleys)>=150)[1]]
step_points_init = np.insert(step_points_init,0,step_valleys[0][0])
step_points_end = step_valleys[:,np.where(np.diff(step_valleys)>=150)[1]+1]
step_points_end  = np.insert(step_points_end,np.shape(step_points_end)[1],step_valleys[0][-1])

Fz_l= np.asarray([x*(-1) for x in Fz2[:18000]])
Fz_l= Fz_l[Fz_l >= 0]
step_valleys_l = np.asarray(np.where(Fz_l==0))
step_points_init_l = step_valleys_l[:,np.where(np.diff(step_valleys_l)>=150)[1]]
step_points_init_l = np.insert(step_points_init_l,0,step_valleys_l[0][0])
step_points_end_l = step_valleys_l[:,np.where(np.diff(step_valleys_l)>=150)[1]+1]
step_points_end_l = np.insert(step_points_end_l,np.shape(step_points_end_l)[1],step_valleys_l[0][-1])
gait_cycles =  np.vstack((step_points_init,step_points_end_l[:42]))
gait_cycles2 = np.sort(np.hstack((step_points_init,step_points_end_l[:42])))

# plot step init
#plt.scatter(step_points_init_l,np.zeros(len(step_points_init_l)))
#plt.scatter(step_points_end_l,np.zeros(len(step_points_end_l)))
plt.plot(Fz_r)
plt.plot(Fz_l)
plt.scatter(gait_cycles2,np.zeros(len(gait_cycles2)))

# Animation 
#def animate(i):
#    scat.set_offsets(np.c_[y[i],z[i]])
#    return scat
#    
#fig, ax =  plt.subplots()
#scat = ax.scatter(y[0], z[0], c = y[0])
#ax.set_xlim([-1000, 1000])
#
#anim = animation.FuncAnimation(fig, animate, interval=50)
# 
#plt.draw()
#plt.show()
#
## to save animation
##FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
##anim.save("sagital.mp4", writer=FFwriter)
#fig, ax =  plt.subplots()
#ax.scatter(y[0],z[0])
#ax.set_xlim([-1000, 1000])
#line = Line2D(y[0], z[0])
#ax.add_line(line)
#plt.show()
##plt.plot(a_l_knee_angle[:1000])
##plt.plot(r_thigh_angle[:1000])
##plt.plot(r_shank_angle[:1000])
##plt.plot(r_hip_angle[:1000])
#plt.figure(1)
#plt.plot(Fz1[8500:9500])
#plt.plot([x*(-1) for x in Fz2[8500:9500]])
#plt.figure(2)
#plt.plot(a_r_knee_angle[8500:9500])
#plt.plot(a_l_knee_angle[8500:9500])
##plt.legend([0a,b,c,d],['0','1','2','3'], loc='upper left')
#plt.show()

#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(Fz1[8500:9500])
#ax1.plot([x*(-1) for x in Fz2[8500:9500]])
#ax1.set_ylabel('Fz1, Fz2')
#
#ax2 = ax1.twinx()
#ax2.plot(r_hip_angle[8500:9500], 'r-')
#ax2.plot(l_hip_angle[8500:9500], 'g-')
#ax2.set_ylabel('right-left hip', color='r')
#for tl in ax2.get_yticklabels():
#    tl.set_color('r')
#    
#plt.show()