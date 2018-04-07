# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import c3d
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA

reader = c3d.Reader(open('/home/gabych/Documents/ETH/gaitIdentification/Vicon/Kai/Level.c3d','rb'))
points = []  # 13 markers, 5 values (x,y,z, residual value, cameras value)
for i, point, analog in reader.read_frames():
    points.append(point)
#  
#[L_TRC, R_TRC, COM, R_KNE, R_UIM, L_KNE, L_UIM, L_ANK, L_LIM, R_ANK, R_LIM, R_MT5, L_MT5]  
x = np.zeros((87500,13,1))
y = np.zeros((87500,13,1))
z = np.zeros((87500,13,1))
for p in range(0,len(points)):
    for j in range(0,len(points[0])):
        x[p][j][:]=points[p][j][0]
        y[p][j][:]=points[p][j][1]
        z[p][j][:]=points[p][j][2]

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
        
        
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x[104000:104013],y[104000:104013],z[104000:104013])
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')




#hip_center = ((x[104000]+x[104001])/2, (z[104000]+z[104001])/2)
#thigh = (x[104004],y[104004],z[104004])
#knee = (x[104003],y[104003],z[104003])
#shank = (x[104010],y[104010],z[104010])
#plt.scatter(hip_center[0],hip_center[1])
#plt.scatter(thigh[0],thigh[2],)
#plt.scatter(knee[0],knee[2])
#plt.scatter(shank[0],shank[2])


a_r_knee_angle = []
a_l_knee_angle = []
b_r_knee_angle = []
b_l_knee_angle = []
r_hip_angle = []
l_hip_angle = []
trunk_angle = []
l_shank_angle = []
r_shank_angle = []
l_thigh_angle = []
r_thigh_angle = []

# angle calculation in sagital plane
for l in range(0,len(R_KNE)):
    l_shank_angle.append(math.degrees(math.atan2(L_KNE[l][2]-L_ANK[l][2],L_KNE[l][1]-L_ANK[l][1])))
    r_shank_angle.append(math.degrees(math.atan2(R_KNE[l][2]-R_ANK[l][2],R_KNE[l][1]-R_ANK[l][1])))
    l_thigh_angle.append(180-math.degrees(math.atan2(L_TRC[l][2]-L_KNE[l][2],L_TRC[l][1]-L_KNE[l][1])))
    r_thigh_angle.append(180-math.degrees(math.atan2(R_TRC[l][2]-R_KNE[l][2],R_TRC[l][1]-R_KNE[l][1])))
    trunk_angle.append(math.degrees(math.atan((C_TRC[l][2]-COM[l][2]/C_TRC[l][1]-COM[l][1]))))
    a_r_knee_angle.append(r_thigh_angle[l]-r_shank_angle[l])
    a_l_knee_angle.append(l_thigh_angle[l]-l_shank_angle[l])
    b_r_knee_angle.append(r_shank_angle[l]+(180-r_thigh_angle[l]))
    b_l_knee_angle.append(l_shank_angle[l]+(180-l_thigh_angle[l]))
    r_hip_angle.append(r_thigh_angle[l]-trunk_angle[l])
    l_hip_angle.append(l_thigh_angle[l]-trunk_angle[l])
    



#k_angle= math.acos(np.dot(p1,p2)/np.dot(LA.norm(p1),LA.norm(p2)))
    #    rk_angle = math.acos(np.dot(R_UIM[l],R_LIM[l])/np.dot(LA.norm(R_UIM[l]),LA.norm(R_LIM[l])))
#    rk_angle = math.atan2(LA.norm(np.cross(R_UIM[l],R_LIM[l])),np.dot(R_UIM[l],R_LIM[l]))
#plt.plot(r_knee_angle)
#plt.plot(l_knee_angle[:1000])
#plt.plot(r_hip_angle[:1000])
#plt.plot(l_hip_angle[:1000])

#plt.scatter(y[0:13], z[0:13])
#plt.scatter(R_UIM[0][0], R_UIM[0][2])
#plt.scatter(R_KNE[0][0], R_KNE[0][2])
#plt.scatter(R_LIM[0][0], R_LIM[0][2])
#plt.scatter(R_UIM[0][0]-R_KNE[0][0], R_UIM[0][2]-R_KNE[0][2])
#plt.scatter(R_LIM[0][0]-R_KNE[0][0], R_LIM[0][2]-R_KNE[0][2])
#plt.xlim((-1000, 1000))
#
#plt.show()

# Animation 
def animate(i):
    scat.set_offsets(np.c_[y[i],z[i]])
    return scat
    
fig, ax =  plt.subplots()
scat = ax.scatter(y[0], z[0], c = y[0])
ax.set_xlim([-1000, 1000])

anim = animation.FuncAnimation(fig, animate, interval=50)
 
plt.draw()
plt.show()

# to save animation
FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
anim.save("sagital.mp4", writer=FFwriter)