import numpy as np
from codac import *
from codac.unsupported import *
import datetime
import cv2
from math import pi,floor,ceil
import matplotlib.pyplot as plt
from collections import deque

def ConcatenateTubes(x,dt):
    ## This function concatenate tubes
    ## x is a list of tubes to be concatenated
    ## it is assumed that x[i](x[i].tdomain().ub()) & x[i+1](x[i+1].tdomain().lb()) != empty
    ## dt is the discretization step
    total_time = 0
    # print("begin")
    for i in range(len(x)):
        total_time += x[i].tdomain().diam()
        # print("total_time = ",total_time)
    # total_time += (len(x)-1)*dt
    res = TubeVector(Interval(x[0].tdomain().lb(),x[0].tdomain().lb()+total_time),dt,2)
    # print(res)
    cmt_shift = 0

    for i in range(x[0][0].nb_slices()):
        t = x[0][0].slice(i).tdomain()
        res[0].set(x[0][0](t),t)
        res[1].set(x[0][1](t),t)
    last_t = Interval(x[0].tdomain().lb())
    for idx in range(len(x)):
        if(idx > 0):
            x[idx].shift_tdomain(cmt_shift)
        for i in np.arange(1,x[idx][0].nb_slices()):
            t = x[idx][0].slice(i).tdomain()
            res[0].set(x[idx][0](t)|x[idx][0](last_t),t|last_t)
            res[1].set(x[idx][1](t)|x[idx][1](last_t),t|last_t)
            last_t = t
        cmt_shift += x[idx].tdomain().diam()

    return res

def InverseTube(x,tdomain,dt):
    res = TubeVector(tdomain,dt,2)
    for i in range(x.nb_slices()):
        # res_i = TubeVector(res[0].slice(i).tdomain(),dt,IntervalVector([x(x[0].slice(x.nb_slices() -1 -i).tdomain())[0],x(x[1].slice(x.nb_slices() -1 -i).tdomain())[1]]))
        res.set(IntervalVector([x(x[0].slice(x.nb_slices() -1 -i).tdomain())[0],x(x[1].slice(x.nb_slices() -1 -i).tdomain())[1]]),res[0].slice(i).tdomain())

    return res

def ContourTube(x_robot,dx_robot,ddx_robot,dt,L):
    ## This function creates the sonar's contour from the robot's state tube
    ## x_robot is a TubeVector with the robot's state (position and orientation)
    ## dx_robot is a TubeVector with the first derivative of the robot's state
    ## dt is the discretization step
    ## dims is a list with the robot's range of visibility on coordinates x and y respectively if dimension == 2 and a list with the lateral distance in coordinates y if dimension == 1
    ## It returns:
    ## a TubeVector with the sonar's contour
    ## v, a list with the first derivative of each part of the sonar's contours

    tdomain = x_robot.tdomain()
    v = [0,0,0,0]

    sin_theta = dx_robot[1]/sqrt(dx_robot[0]*dx_robot[0] + dx_robot[1]*dx_robot[1])
    cos_theta = dx_robot[0]/sqrt(dx_robot[0]*dx_robot[0] + dx_robot[1]*dx_robot[1])
    hip = sqrt(dx_robot[0]*dx_robot[0] + dx_robot[1]*dx_robot[1])
    dhip = (dx_robot[0]*ddx_robot[0] + dx_robot[1]*ddx_robot[1])/hip
    dsin_theta = (ddx_robot[1]*hip - dhip*dx_robot[1])/(hip*hip)
    dcos_theta = (ddx_robot[0]*hip - dhip*dx_robot[0])/(hip*hip)
    
    #right contour
    x_right = TubeVector(tdomain,dt,2)
    v_right = TubeVector(tdomain,dt,2)
    x_right[0] = x_robot[0] + L*sin_theta
    x_right[1] = x_robot[1] - L*cos_theta
    v_right[0] = dx_robot[0] + L*dsin_theta
    v_right[1] = dx_robot[1] - L*dcos_theta
    v[0] = v_right
   
    #left contour
    x_left = TubeVector(tdomain,dt,2)
    v_left = TubeVector(tdomain,dt,2)
    x_left[0] = x_robot[0] - L*sin_theta
    x_left[1] = x_robot[1] + L*cos_theta
    v_left[0] = dx_robot[0] - L*dsin_theta
    v_left[1] = dx_robot[1] + L*dcos_theta
    v[2] = InverseTube(v_left,tdomain,dt)

    alpha0 = (((x_left[0](tdomain.ub()).ub() - x_right[0](tdomain.ub()).ub())/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_right[0](tdomain.ub()).ub())
    alpha1 = (((x_left[1](tdomain.ub()).lb() - x_right[1](tdomain.ub()).lb())/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_right[1](tdomain.ub()).lb())

    d_xrl = TubeVector(tdomain, dt, 2)
    d_xrl[0] &= Interval(alpha0)
    d_xrl[1] &= Interval(alpha1)

    x_rl_ub = TrajectoryVector(2)
    x_rl_ub = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    alpha0 = (((x_left[0](tdomain.ub()).lb() - x_right[0](tdomain.ub()).lb())/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_right[0](tdomain.ub()).lb())
    alpha1 = (((x_left[1](tdomain.ub()).ub() - x_right[1](tdomain.ub()).ub())/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_right[1](tdomain.ub()).ub())

    x_rl_lb = TrajectoryVector(2)
    x_rl_lb = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    alpha0 = (((x_right[0](tdomain.lb()).ub()  - x_left[0](tdomain.lb()).ub() )/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_left[0](tdomain.lb()).ub() )
    alpha1 = (((x_right[1](tdomain.lb()).lb()  - x_left[1](tdomain.lb()).lb())/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_left[1](tdomain.lb()).lb())

    d_xlr = TubeVector(tdomain, dt, 2)
    d_xlr[0] &= Interval(alpha0)
    d_xlr[1] &= Interval(alpha1)

    x_lr_ub = TrajectoryVector(2)
    x_lr_ub = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    alpha0 = (((x_right[0](tdomain.lb()).lb()  - x_left[0](tdomain.lb()).lb() )/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_left[0](tdomain.lb()).lb() )
    alpha1 = (((x_right[1](tdomain.lb()).ub()  - x_left[1](tdomain.lb()).ub())/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_left[1](tdomain.lb()).ub())

    x_lr_lb = TrajectoryVector(2)
    x_lr_lb = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    x_rl = TubeVector(x_rl_lb,x_rl_ub,dt)
    x_lr = TubeVector(x_lr_lb,x_lr_ub,dt)

    x_left = InverseTube(x_left,tdomain,dt)
    
    # pt_1 = ConcatenateTubes([x_right,x_rl],dt)
    # pt_2 = ConcatenateTubes([x_left,x_lr],dt)
    return ConcatenateTubes([x_right,x_rl,x_left,x_lr],dt),v

    # u_real = TubeVector(tdomain, dt, 2)
    # u_real[0] = dx_robot[0]/sqrt((dx_robot[0]*dx_robot[0]) + (dx_robot[1]*dx_robot[1]))
    # u_real[1] = dx_robot[1]/sqrt((dx_robot[1]*dx_robot[1]) + (dx_robot[1]*dx_robot[1]))
    # print("dx_robot= ",dx_robot)
    # #right
    # x_right = TubeVector(tdomain,dt,2)
    # x_right[0] = x_robot[0]+ L*u_real[1]
    # x_right[1] = x_robot[1] - L*u_real[0]

    # #left
    # x_left = TubeVector(tdomain,dt,2)
    # x_left[0] = x_robot[0] - L*u_real[1]
    # x_left[1] = x_robot[1] + L*u_real[0]

    # #right
    # # x_right = TubeVector(tdomain,dt,2)
    # v_right = TubeVector(tdomain,dt,2)
    # # x_right[0] = x_robot[0] + L*sin(x_robot[2])
    # # x_right[1] = x_robot[1] - L*cos(x_robot[2])
    # v_right[0] = dx_robot[0]
    # v_right[1] = dx_robot[1]
    # # v[0] = v_right

    # #left
    # # x_left = TubeVector(tdomain,dt,2)
    # v_left = TubeVector(tdomain,dt,2)
    # # x_left[0] = x_robot[0] - L*sin(x_robot[2])
    # # x_left[1] = x_robot[1] + L*cos(x_robot[2])
    # v_left[0] = -dx_robot[0]
    # v_left[1] = -dx_robot[1]
    # # v[2] = InverseTube(v_left,dt)

    # alpha0 = (((x_left[0](tdomain.ub()).ub() - x_right[0](tdomain.ub()).ub())/(tdomain.ub() - tdomain.lb())))
    # beta0 = (x_right[0](tdomain.ub()).ub())
    # alpha1 = (((x_left[1](tdomain.ub()).lb() - x_right[1](tdomain.ub()).lb())/(tdomain.ub() - tdomain.lb())))
    # beta1 = (x_right[1](tdomain.ub()).lb())

    # d_xrl = TubeVector(tdomain, dt, 2)
    # d_xrl[0] &= Interval(alpha0)
    # d_xrl[1] &= Interval(alpha1)

    # x_rl_ub = TrajectoryVector(2)
    # x_rl_ub = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    # alpha0 = (((x_left[0](tdomain.ub()).lb() - x_right[0](tdomain.ub()).lb())/(tdomain.ub() - tdomain.lb())))
    # beta0 = (x_right[0](tdomain.ub()).lb())
    # alpha1 = (((x_left[1](tdomain.ub()).ub() - x_right[1](tdomain.ub()).ub())/(tdomain.ub() - tdomain.lb())))
    # beta1 = (x_right[1](tdomain.ub()).ub())

    # x_rl_lb = TrajectoryVector(2)
    # x_rl_lb = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    # alpha0 = (((x_right[0](tdomain.lb()).ub()  - x_left[0](tdomain.lb()).ub() )/(tdomain.ub() - tdomain.lb())))
    # beta0 = (x_left[0](tdomain.lb()).ub() )
    # alpha1 = (((x_right[1](tdomain.lb()).lb()  - x_left[1](tdomain.lb()).lb())/(tdomain.ub() - tdomain.lb())))
    # beta1 = (x_left[1](tdomain.lb()).lb())

    # d_xlr = TubeVector(tdomain, dt, 2)
    # d_xlr[0] &= Interval(alpha0)
    # d_xlr[1] &= Interval(alpha1)

    # x_lr_ub = TrajectoryVector(2)
    # x_lr_ub = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    # alpha0 = (((x_right[0](tdomain.lb()).lb()  - x_left[0](tdomain.lb()).lb() )/(tdomain.ub() - tdomain.lb())))
    # beta0 = (x_left[0](tdomain.lb()).lb() )
    # alpha1 = (((x_right[1](tdomain.lb()).ub()  - x_left[1](tdomain.lb()).ub())/(tdomain.ub() - tdomain.lb())))
    # beta1 = (x_left[1](tdomain.lb()).ub())

    # x_lr_lb = TrajectoryVector(2)
    # x_lr_lb = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    # # x_lr = TrajectoryVector(2)
    # # x_lr = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    # x_rl = TubeVector(x_rl_lb,x_rl_ub,dt)
    # x_lr = TubeVector(x_lr_lb,x_lr_ub,dt)


    # return ConcatenateTubes([x_right,x_rl,InverseTube(x_left,tdomain,dt),x_lr],dt),[v_right,d_xrl,InverseTube(v_left,tdomain,dt),d_xlr]

def ConcatenateTraj(x,dt):
    ## This function concatenate trajectories
    ## x is a list of trajectories to be concatenated
    ## it is assumed that x[i](x[i].tdomain().ub()) == x[i+1](x[i+1].tdomain().lb())
    ## dt is the discretization step
    map_x = x[0][0].sample(dt).sampled_map()
    map_y = x[0][1].sample(dt).sampled_map()
    cmt_shift = x[0].tdomain().diam() #cumulative time to shift

    for i in np.arange(1,len(x)):
        x[i] = x[i].sample(dt)
        x[i].shift_tdomain(cmt_shift)

        map_x.update(x[i][0].sampled_map())
        map_y.update(x[i][1].sampled_map())

        cmt_shift += x[i].tdomain().diam()

    res = TrajectoryVector(2)
    res[0] = Trajectory(map_x)
    res[1] = Trajectory(map_y)
    return res

def InverseTraj(x,dt):
    ## This function inverses trajectory x
    ## dt is the discretization step
    x_ = x.sample(dt)
    map_x = x_[0].sampled_map()
    map_y = x_[1].sampled_map()
    keys_x = list(map_x)
    size_lt = len(keys_x)

    for i in range(int((size_lt - (size_lt%2))/2)):
        aux = map_x[keys_x[size_lt - 1 -i]]
        map_x[keys_x[size_lt - 1 -i]] = map_x[keys_x[i]]
        map_x[keys_x[i]] = aux
        aux = map_y[keys_x[size_lt - 1 -i]]
        map_y[keys_x[size_lt - 1 -i]] = map_y[keys_x[i]]
        map_y[keys_x[i]] = aux

    res = TrajectoryVector(2)
    res[0] = Trajectory(map_x)
    res[1] = Trajectory(map_y)
    return res

def ContourTraj(x_robot,dx_robot,ddx_robot,dt,L):
    ## This function creates the sonar's contour from the robot's trajectory
    ## x_robot is a TrajectoryVector with the robot's state (position and orientation)
    ## dx_robot is a TrajectoryVector with the first derivative of the robot's state
    ## dt is the discretization step
    ## d is the lateral distance in coordinate y if dimension == 1
    ## It returns:
    ## the sonar's contour
    ## v, a list with the first derivative of each part of the sonar's contours

    tdomain = x_robot.tdomain()
    v = [0,0,0,0]

    sin_theta = dx_robot[1].sample(dt)/sqrt(dx_robot[0].sample(dt)*dx_robot[0].sample(dt) + dx_robot[1].sample(dt)*dx_robot[1].sample(dt))
    cos_theta = dx_robot[0].sample(dt)/sqrt(dx_robot[0].sample(dt)*dx_robot[0].sample(dt) + dx_robot[1].sample(dt)*dx_robot[1].sample(dt))
    hip = sqrt(dx_robot[0]*dx_robot[0] + dx_robot[1]*dx_robot[1])
    dhip = (dx_robot[0]*ddx_robot[0] + dx_robot[1]*ddx_robot[1])/hip
    dsin_theta = (ddx_robot[1]*hip - dhip*dx_robot[1])/(hip*hip)
    dcos_theta = (ddx_robot[0]*hip - dhip*dx_robot[0])/(hip*hip)

    #right contour
    x_right = TrajectoryVector(2)
    v_right = TrajectoryVector(2)
    x_right[0] = x_robot[0] + L*sin_theta
    x_right[1] = x_robot[1] - L*cos_theta
    v_right[0] = dx_robot[0] + L*dsin_theta
    v_right[1] = dx_robot[1] - L*dcos_theta
    v[0] = v_right
   
    #left contour
    x_left = TrajectoryVector(2)
    v_left = TrajectoryVector(2)
    x_left[0] = x_robot[0] - L*sin_theta
    x_left[1] = x_robot[1] + L*cos_theta
    v_left[0] = dx_robot[0] - L*dsin_theta
    v_left[1] = dx_robot[1] + L*dcos_theta
    v[2] = InverseTraj(v_left,dt)
    
    #right to left
    alpha0 = (((x_left[0](tdomain.ub()) - x_right[0](tdomain.ub()))/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_right[0](tdomain.ub()))
    alpha1 = (((x_left[1](tdomain.ub()) - x_right[1](tdomain.ub()))/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_right[1](tdomain.ub()))
    d_xrl = TrajectoryVector(np.arange(tdomain.lb(),tdomain.ub()+dt,dt), np.array([np.ones((1, len(np.arange(tdomain.lb(),tdomain.ub()+dt,dt)))), np.ones((1, len(np.arange(tdomain.lb(),tdomain.ub()+dt,dt))))]))
    d_xrl[0] *= alpha0
    d_xrl[1] *= alpha1
    x_rl = TrajectoryVector(2)
    x_rl = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))
    v[1] = d_xrl

    #left to right
    alpha0 = (((x_right[0](tdomain.lb()) - x_left[0](tdomain.lb()))/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_left[0](tdomain.lb()))
    alpha1 = (((x_right[1](tdomain.lb()) - x_left[1](tdomain.lb()))/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_left[1](tdomain.lb()))
    d_xlr = TrajectoryVector(np.arange(tdomain.lb(),tdomain.ub()+dt,dt), np.array([np.ones((1, len(np.arange(tdomain.lb(),tdomain.ub()+dt,dt)))), np.ones((1, len(np.arange(tdomain.lb(),tdomain.ub()+dt,dt))))]))
    d_xlr[0] *= alpha0
    d_xlr[1] *= alpha1
    x_lr = TrajectoryVector(2)
    x_lr = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))
    v[3] = d_xlr
    return ConcatenateTraj([x_right,x_rl,InverseTraj(x_left,dt),x_lr],dt),v

def ContourRL(x_right,dx_right,x_left,dx_left,dt):
    ## This function creates the sonar's contour from the right and left extremity trajectories
    ## x_right is a TrajectoryVector with the sonar's extremity on the right
    ## dx_right is a TrajectoryVector with the first derivative of x_right
    ## x_left is a TrajectoryVector with the sonar's extremity on the left
    ## dx_left is a TrajectoryVector with the first derivative of x_left
    ## dt is the discretization step
    ## It returns:
    ## the sonar's contour
    ## v, a list with the first derivative of each part of the sonar's contours

    tdomain = x_right.tdomain()
    v = [0,0,0,0]

    #right contour
    v[0] = dx_right
    #left contour
    v[2] = InverseTraj(dx_left,dt)
    #right to left
    t_values = np.arange(tdomain.lb(),tdomain.ub()+dt,dt)
    alpha0 = (((x_left[0](tdomain.ub()) - x_right[0](tdomain.ub()))/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_right[0](tdomain.ub()))
    alpha1 = (((x_left[1](tdomain.ub()) - x_right[1](tdomain.ub()))/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_right[1](tdomain.ub()))
    d_xrl = TrajectoryVector(t_values, np.array([np.ones((1, len(t_values))), np.ones((1, len(t_values)))]))
    d_xrl[0] *= alpha0
    d_xrl[1] *= alpha1
    x_rl = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))
    v[1] = d_xrl

    #left to right
    t_values = np.arange(tdomain.lb(),tdomain.ub(),dt)
    alpha0 = (((x_right[0](tdomain.lb()) - x_left[0](tdomain.lb()))/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_left[0](tdomain.lb()))
    alpha1 = (((x_right[1](tdomain.lb()) - x_left[1](tdomain.lb()))/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_left[1](tdomain.lb()))
    d_xlr = TrajectoryVector(t_values, np.array([np.ones((1, len(t_values))), np.ones((1, len(t_values)))]))
    d_xlr[0] *= alpha0
    d_xlr[1] *= alpha1
    x_lr = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))
    v[3] = d_xlr

    return ConcatenateTraj([x_right,x_rl,InverseTraj(x_left,dt),x_lr],dt),v

def GammaPlus(dt,x_truth,dx_robot,ddx_robot,L):
    tdomain = x_truth.tdomain()
    sin_theta = dx_robot[1].sample(dt)/sqrt(dx_robot[0].sample(dt)*dx_robot[0].sample(dt) + dx_robot[1].sample(dt)*dx_robot[1].sample(dt))
    cos_theta = dx_robot[0].sample(dt)/sqrt(dx_robot[0].sample(dt)*dx_robot[0].sample(dt) + dx_robot[1].sample(dt)*dx_robot[1].sample(dt))
    hip = sqrt(dx_robot[0]*dx_robot[0] + dx_robot[1]*dx_robot[1])
    dhip = (dx_robot[0]*ddx_robot[0] + dx_robot[1]*ddx_robot[1])/hip

    dsin_theta = (ddx_robot[1]*hip - dhip*dx_robot[1])/(hip*hip)
    dcos_theta = (ddx_robot[0]*hip - dhip*dx_robot[0])/(hip*hip)

    lr = (dx_robot[1]*sin_theta + dx_robot[0]*cos_theta)/(sin_theta*dcos_theta - cos_theta*dsin_theta)
    ll = (dx_robot[1]*sin_theta + dx_robot[0]*cos_theta)/(-sin_theta*dcos_theta + cos_theta*dsin_theta)
    ll_dict = dict()
    lr_dict = dict()

    y_mosaic = []
    yt_right = []
    yt_left = []
    new_back_l = 0
    new_back_r = 0
    t = tdomain.lb() - dt

    while t < tdomain.ub():
        t += dt
        if(t < tdomain.lb()):
            t = tdomain.lb()
        if(t > tdomain.ub()):
            t = tdomain.ub()
        
        if(ll(t) > L):
            ll_dict[t] = L
            new_back_l = 0
        elif(ll(t) < 0):
            ll_dict[t] =L
            new_back_l = 0
        else:
            ll_dict[t] = ll(t)
            if(not new_back_l):
                new_back_l = 1
                yt_left.append(Interval(t).inflate(dt))
            else:
                yt_left[-1] = yt_left[-1] | Interval(t).inflate(dt)
        
        if(lr(t) > L):
            lr_dict[t] = L
            new_back_r = 0
        elif(lr(t) < 0):
            lr_dict[t] = L
            new_back_r = 0
        else:
            lr_dict[t] = lr(t)
            if(not new_back_r):
                new_back_r = 1
                yt_right.append(Interval(t).inflate(dt))
            else:
                yt_right[-1] = yt_right[-1] | Interval(t).inflate(dt)

    lr = Trajectory(lr_dict)
    ll = Trajectory(ll_dict)

    tan_traj_r = TrajectoryVector(2)
    dtan_traj_r = TrajectoryVector(2)
    tan_traj_l = TrajectoryVector(2)
    dtan_traj_l = TrajectoryVector(2)

    tdomain = lr.tdomain() #had to this to work with mac new processors
    tan_traj_r[0] = x_truth[0].truncate_tdomain(tdomain) + lr*sin_theta.truncate_tdomain(tdomain)
    tan_traj_r[1] = x_truth[1].truncate_tdomain(tdomain)- lr*cos_theta.truncate_tdomain(tdomain)
    tan_traj_l[0] = x_truth[0].truncate_tdomain(tdomain) - ll*sin_theta.truncate_tdomain(tdomain)
    tan_traj_l[1] = x_truth[1].truncate_tdomain(tdomain) + ll*cos_theta.truncate_tdomain(tdomain)

    data_vx_r = []
    data_vy_r = []
    data_vx_l = []
    data_vy_l = []
    t = [tdomain.lb()]
    while(t[-1] < tan_traj_r.tdomain().ub()):
        t_aux = t[-1]+dt
        if(t_aux > tdomain.ub()):
            t_aux = tdomain.ub()
   
        v0_r =  (tan_traj_r[0](t_aux) - tan_traj_r[0](t[-1]))/dt
        v1_r =  (tan_traj_r[1](t_aux) - tan_traj_r[1](t[-1]))/dt
        v0_l =  (tan_traj_l[0](t_aux) - tan_traj_l[0](t[-1]))/dt
        v1_l =  (tan_traj_l[1](t_aux) - tan_traj_l[1](t[-1]))/dt
        data_vx_r.append(v0_r)
        data_vy_r.append(v1_r)
        data_vx_l.append(v0_l)
        data_vy_l.append(v1_l)
        t.append(t_aux)
    
    data_vx_r.append(data_vx_r[-1])
    data_vy_r.append(data_vy_r[-1])
    data_vx_l.append(data_vx_l[-1])
    data_vy_l.append(data_vy_l[-1])
    dtan_traj_r[0] = Trajectory(t, data_vx_r)
    dtan_traj_r[1] = Trajectory(t, data_vy_r)
    dtan_traj_l[0] = Trajectory(t, data_vx_l)
    dtan_traj_l[1] = Trajectory(t, data_vy_l)


    gamma_after,v_after = ContourRL(tan_traj_r,dtan_traj_r,tan_traj_l,dtan_traj_l,dt)

    return gamma_after,v_after,yt_right,yt_left



def TangentLoop(v,tdomain,loops):
    d_list_i = []
    d_list_f = []

    for l in loops:
        if(l[0].ub() <= tdomain.ub()): #right
            v_begin_x = v[0](l[0])[0]
            v_begin_y = v[0](l[0])[1]
        elif(l[0].ub() <= tdomain.ub() + tdomain.diam()): #rl
            v_begin_x = v[1](l[0]-tdomain.diam())[0]
            v_begin_y = v[1](l[0]-tdomain.diam())[1]
        elif(l[0].ub() <= tdomain.ub() + 2*tdomain.diam()): #left
            v_begin_x = -v[2]((l[0]-2*tdomain.diam()))[0]
            v_begin_y = -v[2]((l[0]-2*tdomain.diam()))[1]
        else: #lr
            v_begin_x = v[3](l[0]-3*tdomain.diam())[0]
            v_begin_y = v[3](l[0]-3*tdomain.diam())[1]

        if(l[1].ub() <= tdomain.ub()): #right
            v_end_x = v[0](l[1])[0]
            v_end_y = v[0](l[1])[1]
        elif(l[1].ub() <= tdomain.ub() + tdomain.diam()): #rl
            v_end_x = v[1](l[1]-tdomain.diam())[0]
            v_end_y = v[1](l[1]-tdomain.diam())[1]
        elif(l[1].ub() <= tdomain.ub() + 2*tdomain.diam()): #left
            v_end_x = -v[2]((l[1]-2*tdomain.diam()))[0]
            v_end_y = -v[2]((l[1]-2*tdomain.diam()))[1]
        else: #lr
            v_end_x = v[3](l[1]-3*tdomain.diam())[0]
            v_end_y = v[3](l[1]-3*tdomain.diam())[1]

        d_list_i.append([v_begin_x,v_begin_y])
        d_list_f.append([v_end_x,v_end_y])
    
    return d_list_i,d_list_f