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
L_TRC = np.zeros((len(points),3))
R_TRC = np.zeros((len(points),3))
C_TRC = np.zeros((len(points),3))
COM = np.zeros((len(points),3))
R_KNE = np.zeros((len(points),3))
R_UIM = np.zeros((len(points),3))
L_KNE = np.zeros((len(points),3))
L_UIM = np.zeros((len(points),3))
L_ANK = np.zeros((len(points),3))
L_LIM = np.zeros((len(points),3))
R_ANK = np.zeros((len(points),3))
R_LIM = np.zeros((len(points),3))
R_MT5 = np.zeros((len(points),3))
L_MT5 = np.zeros((len(points),3))

for p in range(0,len(points)):
    L_TRC[p,:] = points[p][0][0:3]
    R_TRC[p,:] = points[p][1][0:3]
    C_TRC[p,:] = ((points[p][1][0:3]+points[p][0][0:3])/2)
    COM[p,:] = points[p][2][0:3]
    R_KNE[p,:] = points[p][3][0:3]
    R_UIM[p,:] = points[p][4][0:3]
    L_KNE[p,:] = points[p][5][0:3]
    L_UIM[p,:] = points[p][6][0:3]
    L_ANK[p,:] = points[p][7][0:3]
    L_LIM[p,:] = points[p][8][0:3]
    R_ANK[p,:] = points[p][9][0:3]
    R_LIM[p,:] = points[p][10][0:3]
    R_MT5[p,:] = points[p][11][0:3]
    L_MT5[p,:] = points[p][12][0:3]
        
# joint angle calculation
a_r_knee_angle = np.zeros((len(points),1))
a_l_knee_angle = np.zeros((len(points),1))
b_r_knee_angle = np.zeros((len(points),1))
b_l_knee_angle = np.zeros((len(points),1))
r_hip_angle = np.zeros((len(points),1))
l_hip_angle = np.zeros((len(points),1))
trunk_angle = np.zeros((len(points),1))
l_shank_angle = np.zeros((len(points),1))
r_shank_angle = np.zeros((len(points),1))
l_thigh_angle = np.zeros((len(points),1))
r_thigh_angle = np.zeros((len(points),1))
r_foot_angle = np.zeros((len(points),1))
l_foot_angle = np.zeros((len(points),1))
r_ankle_angle = np.zeros((len(points),1))
l_ankle_angle = np.zeros((len(points),1))

# angle calculation in sagital plane
for l in range(0,len(R_KNE)):
    l_shank_angle[l] = np.rad2deg(np.arctan2((L_KNE[l][2]-L_ANK[l][2]), (L_KNE[l][1]-L_ANK[l][1]))) # % (2 * np.pi)
    r_shank_angle[l] = np.rad2deg(np.arctan2((R_KNE[l][2]-R_ANK[l][2]), (R_KNE[l][1]-R_ANK[l][1]))) 
    l_thigh_angle[l] = 180-np.rad2deg(np.arctan2((L_TRC[l][2]-L_KNE[l][2]), (L_TRC[l][1]-L_KNE[l][1]))) 
    r_thigh_angle[l] = 180-np.rad2deg(np.arctan2((R_TRC[l][2]-R_KNE[l][2]), (R_TRC[l][1]-R_KNE[l][1]))) 
    trunk_angle[l] = np.rad2deg(np.arctan2((C_TRC[l][2]-COM[l][2]),(C_TRC[l][1]-COM[l][1])))  
    r_foot_angle[l] = np.rad2deg(np.arctan2((R_ANK[l][2]-R_MT5[l][2]),(R_ANK[l][1]-R_MT5[l][1])))
    l_foot_angle[l] = np.rad2deg(np.arctan2((L_ANK[l][2]-L_MT5[l][2]),(L_ANK[l][1]-L_MT5[l][1])))
    a_r_knee_angle[l] = (r_thigh_angle[l]-r_shank_angle[l])
    a_l_knee_angle[l] = (l_thigh_angle[l]-l_shank_angle[l])
    b_r_knee_angle[l] = (r_shank_angle[l]+(180-r_thigh_angle[l]))
    b_l_knee_angle[l] = (l_shank_angle[l]+(180-l_thigh_angle[l]))
    r_hip_angle[l] = (r_thigh_angle[l]-trunk_angle[l])
    l_hip_angle[l] = (l_thigh_angle[l]-trunk_angle[l])
    r_ankle_angle[l] = (r_foot_angle[l]-r_shank_angle[l]-90)
    l_ankle_angle[l] = (l_foot_angle[l]-l_shank_angle[l]-90)

# segments
xc = np.array((COM[:,0],C_TRC[:,0], R_TRC[:,0], L_TRC[:,0])).T
yc = np.array((COM[:,1],C_TRC[:,1], R_TRC[:,1], L_TRC[:,1])).T
zc = np.array((COM[:,2],C_TRC[:,2], R_TRC[:,2], L_TRC[:,2])).T

xr = np.array((R_TRC[:,0], R_UIM[:,0], R_KNE[:,0], R_LIM[:,0], R_ANK[:,0], R_MT5[:,0])).T
yr = np.array((R_TRC[:,1], R_UIM[:,1], R_KNE[:,1], R_LIM[:,1], R_ANK[:,1], R_MT5[:,1])).T
zr = np.array((R_TRC[:,2], R_UIM[:,2], R_KNE[:,2], R_LIM[:,2], R_ANK[:,2], R_MT5[:,2])).T

xl = np.array((L_TRC[:,0], L_UIM[:,0], L_KNE[:,0], L_LIM[:,0], L_ANK[:,0], L_MT5[:,0])).T
yl = np.array((L_TRC[:,1], L_UIM[:,1], L_KNE[:,1], L_LIM[:,1], L_ANK[:,1], L_MT5[:,1])).T
zl = np.array((L_TRC[:,2], L_UIM[:,2], L_KNE[:,2], L_LIM[:,2], L_ANK[:,2], L_MT5[:,2])).T

# text
xt = np.array((R_TRC[:,0], L_TRC[:,0], R_KNE[:,0], L_KNE[:,0], L_ANK[:,0], R_ANK[:,0], COM[:,0])).T
yt = np.array((R_TRC[:,1], L_TRC[:,1], R_KNE[:,1], L_KNE[:,1], L_ANK[:,1], R_ANK[:,1], COM[:,1])).T
zt = np.array((R_TRC[:,2], L_TRC[:,2], R_KNE[:,2], L_KNE[:,2], L_ANK[:,2], R_ANK[:,2], COM[:,2])).T
angles_text = np.array((r_hip_angle[:,0], l_hip_angle[:,0], a_r_knee_angle[:,0], a_l_knee_angle[:,0], l_ankle_angle[:,0], r_ankle_angle[:,0], trunk_angle[:,0])).T
    
# gait cycles from grf
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
gait_cycles =  np.vstack((step_points_init,step_points_end_l[:42])) # first row init points 2nd row end points
gait_cycles2 = np.sort(np.hstack((step_points_init,step_points_end_l[:42])))

# plot step init
#plt.scatter(step_points_init_l,np.zeros(len(step_points_init_l)))
#plt.scatter(step_points_end_l,np.zeros(len(step_points_end_l)))
plt.plot(Fz_r)
plt.plot(Fz_l)
plt.scatter(gait_cycles2,np.zeros(len(gait_cycles2)))

# Animation 
def animate(i):
    scat.set_offsets(np.c_[y[i],z[i]])
    line_c.set_data(yc[i],zc[i])
    line_r.set_data(yr[i],zr[i])
    line_l.set_data(yl[i],zl[i])
    h_r_text.set_text(np.array2string(angles_text[i,0]))
    h_r_text.set_position((yt[i,0]+.8,zt[i,0]))
    h_l_text.set_text(np.array2string(angles_text[i,1]))
    h_l_text.set_position((yt[i,1]+.8,zt[i,1]+.5))
    r_k_text.set_text(np.array2string(angles_text[i,2]))
    r_k_text.set_position((yt[i,2]+.8,zt[i,2]))
    l_k_text.set_text(np.array2string(angles_text[i,3]))
    l_k_text.set_position((yt[i,3]+.8,zt[i,3]))
    l_a_text.set_text(np.array2string(angles_text[i,4]))
    l_a_text.set_position((yt[i,4]+.8,zt[i,4]))
    r_a_text.set_text(np.array2string(angles_text[i,5]))
    r_a_text.set_position((yt[i,5]+.8,zt[i,5]))
    t_text.set_text(np.array2string(angles_text[i,6]))
    t_text.set_position((yt[i,6]+.8,zt[i,6]))
    return scat, line_c, line_r, line_l, h_r_text, h_l_text, r_k_text, l_k_text, l_a_text, r_a_text, t_text
    
fig, ax =  plt.subplots()
scat = ax.scatter(y[0], z[0], c = y[0])
line_c = Line2D(yc[0], zc[0])
line_r = Line2D(yr[0], zr[0])
line_l = Line2D(yl[0], zl[0])
h_r_text = ax.text(yt[0,0]+.8,zt[0,0]+.5, np.array2string(angles_text[0,0]))
h_l_text = ax.text(yt[0,1]+.8,zt[0,1], np.array2string(angles_text[0,1]))
r_k_text = ax.text(yt[0,2]+.8,zt[0,2], np.array2string(angles_text[0,2]))
l_k_text = ax.text(yt[0,3]+.8,zt[0,3], np.array2string(angles_text[0,3]))
l_a_text = ax.text(yt[0,4]+.8,zt[0,4], np.array2string(angles_text[0,4]))
r_a_text = ax.text(yt[0,5]+.8,zt[0,5], np.array2string(angles_text[0,5]))
t_text = ax.text(yt[0,6]+.8,zt[0,6], np.array2string(angles_text[0,6]))
ax.set_xlim([-1000, 1000])
ax.add_line(line_c)
ax.add_line(line_r)
ax.add_line(line_l)

anim = animation.FuncAnimation(fig, animate, interval=100)
 
plt.draw()
plt.show()

# to save animation
FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
#anim.save("sagital.mp4", writer=FFwriter)


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