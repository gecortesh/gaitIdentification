# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:21:11 2018

@author: Gabriela Cortés
"""
import c3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D   
from scipy.signal import butter, filtfilt

# reading data from vicon, return coordinates x,y,z
def read_file(experiment, subject):
    #reader = c3d.Reader(open('/home/gabych/Documents/ETH/gaitIdentification/Vicon/'+ subject +'/'+ experiment +'.c3d','rb'))
    reader = c3d.Reader(open('/cluster/home/corteshg/Subjects/'+ subject +'/'+ experiment +'.c3d','rb'))
    points = []  # 13 markers, 5 values (x,y,z, residual value(estimate of the error for this particular point), cameras value(number of cameras that observed the point))
    vgrf = [] # vertical ground reaction force
    for i, point, analog in reader.read_frames():
        points.append(point)
        vgrf.append(analog)
    return vgrf, points

# saving markers coordinates for visualiztion per frame 
def markers_xyz(points):  
    x = np.zeros((len(points),13,1))
    y = np.zeros((len(points),13,1))
    z = np.zeros((len(points),13,1))
    for p in range(0,len(points)):
        for j in range(0,13):
            x[p][j][:]=points[p][j][0]
            y[p][j][:]=-1*points[p][j][1]
            z[p][j][:]=points[p][j][2]
    return x,y,z
        
# to save the analog data in the corresponding value (Forces and moments per plane and per force plate)
def kinetics(vgrf):
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
    return Fz1, Fz2
        
def markers_coordinates(points):
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
        
    return COM, C_TRC, R_TRC, L_TRC, R_UIM, L_UIM, R_LIM, L_LIM, R_ANK, L_ANK, R_MT5, L_MT5, R_KNE, L_KNE
            
def kinematics(points):
    COM, C_TRC, R_TRC, L_TRC, R_UIM, L_UIM, R_LIM, L_LIM, R_ANK, L_ANK, R_MT5, L_MT5, R_KNE, L_KNE = markers_coordinates(points)
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
    l_shank_angle_v= np.zeros((len(points),1))
    r_shank_angle_v = np.zeros((len(points),1))
    l_thigh_angle_v = np.zeros((len(points),1))
    r_thigh_angle_v = np.zeros((len(points),1))
    r_foot_angle = np.zeros((len(points),1))
    l_foot_angle = np.zeros((len(points),1))
    r_ankle_angle = np.zeros((len(points),1))
    l_ankle_angle = np.zeros((len(points),1))
    
    # angle calculation in sagital plane (l-left, r-right)
    for l in range(0,len(R_KNE)):
        l_shank_angle[l] = np.rad2deg(np.arctan2((L_KNE[l][2]-L_ANK[l][2]), (-1*(L_KNE[l][1]-L_ANK[l][1])))) #+ 360) % 360 # % (2 * np.pi)
        r_shank_angle[l] = np.rad2deg(np.arctan2((R_KNE[l][2]-R_ANK[l][2]), (-1*(R_KNE[l][1]-R_ANK[l][1])))) 
        r_shank_angle_v[l] = np.rad2deg(np.arctan2((R_ANK[l][2]-R_KNE[l][2]),(-1*(R_ANK[l][1]-R_KNE[l][1]))))
        l_shank_angle_v[l] = np.rad2deg(np.arctan2((L_ANK[l][2]-L_KNE[l][2]),(-1*(L_ANK[l][1]-L_KNE[l][1])))) #+ 360) % 360
        l_thigh_angle[l] = np.rad2deg(np.arctan2((L_TRC[l][2]-L_KNE[l][2]), (-1*(L_TRC[l][1]-L_KNE[l][1])))) #+ 360) % 360
        r_thigh_angle[l] = np.rad2deg(np.arctan2((R_TRC[l][2]-R_KNE[l][2]), (-1*(R_TRC[l][1]-R_KNE[l][1])))) 
        l_thigh_angle_v[l] = np.rad2deg(np.arctan2((L_KNE[l][2]-L_TRC[l][2]), (-1*(L_KNE[l][1]-L_TRC[l][1]))))# + 360) % 360
        r_thigh_angle_v[l] = np.rad2deg(np.arctan2((R_KNE[l][2]-R_TRC[l][2]), (-1*(R_KNE[l][1]-R_TRC[l][1]))))# + 360) % 360
        trunk_angle[l] = np.rad2deg(np.arctan2((COM[l][2]-C_TRC[l][2]),(-1*(COM[l][1]-C_TRC[l][1])))) #+ 360) % 360  
        r_foot_angle[l] = np.rad2deg(np.arctan2((R_ANK[l][2]-R_MT5[l][2]),(-1*(R_ANK[l][1]-R_MT5[l][1])))) #+ 360) % 360
        l_foot_angle[l] = np.rad2deg(np.arctan2((L_ANK[l][2]-L_MT5[l][2]),(-1*(L_ANK[l][1]-L_MT5[l][1])))) #+ 360) % 360
        a_r_knee_angle[l] = (r_thigh_angle[l]-r_shank_angle[l])
        a_l_knee_angle[l] = (l_thigh_angle[l]-l_shank_angle[l])
        b_r_knee_angle[l] = (r_shank_angle[l]+(180-r_thigh_angle[l]))
        b_l_knee_angle[l] = (l_shank_angle[l]+(180-l_thigh_angle[l]))
        r_hip_angle[l] = (r_thigh_angle[l]-trunk_angle[l])
        l_hip_angle[l] = (l_thigh_angle[l]-trunk_angle[l])
        r_ankle_angle[l] = (l_foot_angle[l]-l_shank_angle[l])-90
        l_ankle_angle[l] = (l_foot_angle[l]-l_shank_angle[l])-90
    
    return a_r_knee_angle, a_l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle
        
# segments for animation
def segments_anim(points):
    COM, C_TRC, R_TRC, L_TRC, R_UIM, L_UIM, R_LIM, L_LIM, R_ANK, L_ANK, R_MT5, L_MT5, R_KNE, L_KNE = markers_coordinates(points)  
    a_r_knee_angle, a_l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = kinematics(points)
    
    xc = np.array((COM[:,0],C_TRC[:,0], R_TRC[:,0], L_TRC[:,0])).T
    yc = -1*np.array((COM[:,1],C_TRC[:,1], R_TRC[:,1], L_TRC[:,1])).T
    zc = np.array((COM[:,2],C_TRC[:,2], R_TRC[:,2], L_TRC[:,2])).T
    
    xr = np.array((R_TRC[:,0], R_UIM[:,0], R_KNE[:,0], R_LIM[:,0], R_ANK[:,0], R_MT5[:,0])).T
    yr = -1*np.array((R_TRC[:,1], R_UIM[:,1], R_KNE[:,1], R_LIM[:,1], R_ANK[:,1], R_MT5[:,1])).T
    zr = np.array((R_TRC[:,2], R_UIM[:,2], R_KNE[:,2], R_LIM[:,2], R_ANK[:,2], R_MT5[:,2])).T
    
    xl = np.array((L_TRC[:,0], L_UIM[:,0], L_KNE[:,0], L_LIM[:,0], L_ANK[:,0], L_MT5[:,0])).T
    yl = -1*np.array((L_TRC[:,1], L_UIM[:,1], L_KNE[:,1], L_LIM[:,1], L_ANK[:,1], L_MT5[:,1])).T
    zl = np.array((L_TRC[:,2], L_UIM[:,2], L_KNE[:,2], L_LIM[:,2], L_ANK[:,2], L_MT5[:,2])).T
    
    # text
    xt = np.array((R_TRC[:,0], L_TRC[:,0], R_KNE[:,0], L_KNE[:,0], L_ANK[:,0], R_ANK[:,0], COM[:,0])).T
    yt = -1*np.array((R_TRC[:,1], L_TRC[:,1], R_KNE[:,1], L_KNE[:,1], L_ANK[:,1], R_ANK[:,1], COM[:,1])).T
    zt = np.array((R_TRC[:,2], L_TRC[:,2], R_KNE[:,2], L_KNE[:,2], L_ANK[:,2], R_ANK[:,2], COM[:,2])).T
    angles_text = np.array((r_hip_angle[:,0], l_hip_angle[:,0], a_r_knee_angle[:,0], a_l_knee_angle[:,0], l_ankle_angle[:,0], r_ankle_angle[:,0], trunk_angle[:,0])).T
    return xc, yc, zc, xr, yr, zr, xt, yt, zt, xl, yl, zl, angles_text   

# gait cycles from grf
def gait_cycles(vgrf):
    Fz1, Fz2 = kinetics(vgrf)
    Fz_l = butter_lowpass_filter(Fz1,40.0,1000.0,4)
    Fz_l[np.where(Fz_l<=10)] = 0
    step_valleys = np.asarray(np.where(Fz_l==0))
    step_points_init = step_valleys[:,np.where(np.diff(step_valleys)>=100)[1]]
    step_points_init = np.insert(step_points_init,0,step_valleys[0][0])
    step_points_end = step_valleys[:,np.where(np.diff(step_valleys)>=100)[1]+1]
    step_points_end  = np.insert(step_points_end,np.shape(step_points_end)[1],step_valleys[0][-1])
    gait_cycle_l = np.sort(np.hstack((step_points_init,step_points_end)))
    
    Fz_r= butter_lowpass_filter(([x*(-1) for x in Fz2]),40.0,1000.0,4)
    Fz_r[np.where(Fz_r<=10)] = 0
    step_valleys_r = np.asarray(np.where(Fz_r==0))
    step_points_init_r = step_valleys_r[:,np.where(np.diff(step_valleys_r)>=100)[1]]
    step_points_init_r = np.insert(step_points_init_r,0,step_valleys_r[0][0])
    step_points_end_r = step_valleys_r[:,np.where(np.diff(step_valleys_r)>=100)[1]+1]
    step_points_end_r = np.insert(step_points_end_r,np.shape(step_points_end_r)[1],step_valleys_r[0][-1])
    gait_cycle_r = np.sort(np.hstack((step_points_init_r,step_points_end_r)))

    gait_cycles =  np.sort(np.hstack(((gait_cycle_l,gait_cycle_r)))) # first row init points 2nd row end points
    full_gait_cycle =  gait_cycles[0::4]

    #full_gait_cycle = gait_cycles[np.where(np.diff(gait_cycles)>=50)]
    return full_gait_cycle

# plot step vgrf with gait cycles points
    fig0= plt.figure()
    plt.plot(Fz1)
    plt.plot(Fz2)
    fzl, = plt.plot(Fz_l, 'r', label='left')
    fzr, = plt.plot(Fz_r, 'b',  label='right')
    plt.legend([fzl,fzr], ['Left', 'Right'])
    plt.scatter(full_gait_cycle,np.zeros(len(full_gait_cycle)))

def markers_animation(points, name, save):
    xc, yc, zc, xr, yr, zr, xt, yt, zt, xl, yl, zl, angles_text = segments_anim(points)
    x,y,z = markers_xyz(points)
    # Animation 
    def animate(i):
        scat.set_offsets(np.c_[y[i],z[i]])
        line_c.set_data(yc[i],zc[i])
        line_r.set_data(yr[i],zr[i])
        line_l.set_data(yl[i],zl[i])
        h_r_text.set_text(np.array2string(angles_text[i,0]))
        h_r_text.set_position((yt[i,0]+.8,zt[i,0]+1))
        h_l_text.set_text(np.array2string(angles_text[i,1]))
        h_l_text.set_position((yt[i,1]+.8,zt[i,1]))
        r_k_text.set_text(np.array2string(angles_text[i,2]))
        r_k_text.set_position((yt[i,2]+.8,zt[i,2]+1))
        l_k_text.set_text(np.array2string(angles_text[i,3]))
        l_k_text.set_position((yt[i,3]+.8,zt[i,3]))
        l_a_text.set_text(np.array2string(angles_text[i,4]))
        l_a_text.set_position((yt[i,4]+.8,zt[i,4]+1))
        r_a_text.set_text(np.array2string(angles_text[i,5]))
        r_a_text.set_position((yt[i,5]+.8,zt[i,5]))
        t_text.set_text(np.array2string(angles_text[i,6]))
        t_text.set_position((yt[i,6]+.8,zt[i,6]))
        print(i)
        return scat, line_c, line_r, line_l, h_r_text, h_l_text, r_k_text, l_k_text, l_a_text, r_a_text, t_text
        
    fig, ax =  plt.subplots()
    scat = ax.scatter(y[0], z[0], c = y[0])
    line_c = Line2D(yc[0], zc[0])
    line_r = Line2D(yr[0], zr[0])
    line_l = Line2D(yl[0], zl[0])
    h_r_text = ax.text(yt[0,0]+.8,zt[0,0]+1, np.array2string(angles_text[0,0]))
    h_l_text = ax.text(yt[0,1]+.8,zt[0,1], np.array2string(angles_text[0,1]))
    r_k_text = ax.text(yt[0,2]+.8,zt[0,2]+1, np.array2string(angles_text[0,2]))
    l_k_text = ax.text(yt[0,3]+.8,zt[0,3], np.array2string(angles_text[0,3]))
    l_a_text = ax.text(yt[0,4]+.8,zt[0,4]+1, np.array2string(angles_text[0,4]))
    r_a_text = ax.text(yt[0,5]+.8,zt[0,5], np.array2string(angles_text[0,5]))
    t_text = ax.text(yt[0,6]+.8,zt[0,6], np.array2string(angles_text[0,6]))
    ax.set_xlim([-1000, 1000])
    ax.add_line(line_c)
    ax.add_line(line_r)
    ax.add_line(line_l)
    
    anim = animation.FuncAnimation(fig, animate, interval=50)
    plt.draw()
    plt.show()
    
    # to save animation
    if save == True:
        FFwriter = animation.FFMpegWriter(fps=150, extra_args=['-vcodec', 'libx264'])
        anim.save(name + ".mp4", writer=FFwriter)

# plot joint angle "divided" in gait cycles
def angle_gait(experiment, subject, angle):
    vgrf, points =  read_file(experiment, subject)
    full_gait_cycle = gait_cycles(vgrf)
    a_r_knee_angle, a_l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = kinematics(points)
    if angle == 'right knee':
        joint_angle = a_r_knee_angle
    elif angle == 'left knee':
        joint_angle = a_l_knee_angle
    elif angle == 'right hip':
        joint_angle = r_hip_angle
    elif angle == 'left hip':
        joint_angle = l_hip_angle
    elif angle == 'trunk':
        joint_angle = trunk_angle
    elif angle == 'right ankle':
        joint_angle = r_ankle_angle
    elif angle == 'left ankle':
        joint_angle = l_ankle_angle
    else:
        joint_angle = a_l_knee_angle
    
    plt.title(angle)
    plt.plot(joint_angle)
    for l in full_gait_cycle:
        plt.vlines(x=l, ymin=np.min(joint_angle), ymax=np.max(joint_angle))
    plt.show()

# to plot data with different axis limit
def plot_diff_axes(f1, f2):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(f1)
    ax1.set_ylabel('Fz1, Fz2')
    
    ax2 = ax1.twinx()
    ax2.plot(f2, 'r-')
    ax2.set_ylabel('right knee', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')    
    plt.show()

# cyclogram
def cyclogram (a1, a2):
    plt.figure()
    plt.scatter(a1, a2)
    plt.show()

# feature vector calculation per gait cycle
def feature_vector(experiment, subject,w, h):
    vgrf, points =  read_file(experiment, subject)
    a_r_knee_angle, a_l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = kinematics(points)
    full_gait_cycle = gait_cycles(vgrf)
    r_knee_rom = np.zeros((len(full_gait_cycle)-1,1))
    l_knee_rom = np.zeros((len(full_gait_cycle)-1,1))
    r_hip_rom = np.zeros((len(full_gait_cycle)-1,1))
    l_hip_rom = np.zeros((len(full_gait_cycle)-1,1))
    r_ankle_rom = np.zeros((len(full_gait_cycle)-1,1))
    l_ankle_rom = np.zeros((len(full_gait_cycle)-1,1))
    trunk_rom = np.zeros((len(full_gait_cycle)-1,1))
    
    for g in range (0,len(full_gait_cycle)-1):
        r_knee_rom[g] = np.max(a_r_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(a_r_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        l_knee_rom[g] = np.max(a_l_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(a_l_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        r_hip_rom[g] = np.max(r_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(r_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        l_hip_rom[g] = np.max(l_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(l_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        r_ankle_rom[g] = np.max(r_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(r_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        l_ankle_rom[g] = np.max(l_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(l_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        trunk_rom[g] = np.max(trunk_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(trunk_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
    weight = np.ones((len(r_knee_rom),1))*w
    height = np.ones((len(r_knee_rom),1))*h
    x = np.hstack([r_knee_rom, l_knee_rom, r_hip_rom, l_hip_rom, trunk_rom, r_ankle_rom, l_ankle_rom, weight, height])
    return  x

def feature_vector_angles(experiment, subject):
    vgrf, points =  read_file(experiment, subject)
    a_r_knee_angle, a_l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = kinematics(points)
    full_gait_cycle = gait_cycles(vgrf)
    r_knee_rom = np.zeros((len(full_gait_cycle)-1,1))
    l_knee_rom = np.zeros((len(full_gait_cycle)-1,1))
    r_hip_rom = np.zeros((len(full_gait_cycle)-1,1))
    l_hip_rom = np.zeros((len(full_gait_cycle)-1,1))
    r_ankle_rom = np.zeros((len(full_gait_cycle)-1,1))
    l_ankle_rom = np.zeros((len(full_gait_cycle)-1,1))
    trunk_rom = np.zeros((len(full_gait_cycle)-1,1))
    
    for g in range (0,len(full_gait_cycle)-1):
        r_knee_rom[g] = np.max(a_r_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(a_r_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        l_knee_rom[g] = np.max(a_l_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(a_l_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        r_hip_rom[g] = np.max(r_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(r_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        l_hip_rom[g] = np.max(l_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(l_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        r_ankle_rom[g] = np.max(r_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(r_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        l_ankle_rom[g] = np.max(l_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(l_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        trunk_rom[g] = np.max(trunk_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(trunk_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])

    x = np.hstack([r_knee_rom, l_knee_rom, r_hip_rom, l_hip_rom, trunk_rom, r_ankle_rom, l_ankle_rom])
    return  x

def y_vector( experiment, subject, s):
    vgrf, points =  read_file(experiment, subject)
    full_gait_cycle = gait_cycles(vgrf)
    y = np.ones(len(full_gait_cycle)-1)*s
    return y

def feature_clustering(experiment, subject, angle):
    vgrf, points =  read_file(experiment, subject)
    a_r_knee_angle, a_l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = kinematics(points)
    full_gait_cycle = gait_cycles(vgrf)
    r_knee_rom = np.zeros((len(full_gait_cycle)-1,1))
    l_knee_rom = np.zeros((len(full_gait_cycle)-1,1))
    r_hip_rom = np.zeros((len(full_gait_cycle)-1,1))
    l_hip_rom = np.zeros((len(full_gait_cycle)-1,1))
    r_ankle_rom = np.zeros((len(full_gait_cycle)-1,1))
    l_ankle_rom = np.zeros((len(full_gait_cycle)-1,1))
    trunk_rom = np.zeros((len(full_gait_cycle)-1,1))
    
    for g in range (0,len(full_gait_cycle)-1):
        r_knee_rom[g] = np.max(a_r_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(a_r_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        l_knee_rom[g] = np.max(a_l_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(a_l_knee_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        r_hip_rom[g] = np.max(r_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(r_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        l_hip_rom[g] = np.max(l_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(l_hip_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        r_ankle_rom[g] = np.max(r_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(r_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        l_ankle_rom[g] = np.max(l_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(l_ankle_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        trunk_rom[g] = np.max(trunk_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])-np.min(trunk_angle[full_gait_cycle[g]:full_gait_cycle[g+1]])
        
    if angle == 'right knee':
        joint_angle = a_r_knee_angle
    elif angle == 'left knee':
        joint_angle = a_l_knee_angle
    elif angle == 'right hip':
        joint_angle = r_hip_angle
    elif angle == 'left hip':
        joint_angle = l_hip_angle
    elif angle == 'trunk':
        joint_angle = trunk_angle
    elif angle == 'right ankle':
        joint_angle = r_ankle_angle
    elif angle == 'left ankle':
        joint_angle = l_ankle_angle
    else:
        joint_angle = a_l_knee_angle
    return joint_angle

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', output ='ba', analog=False)
    return b,a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

