import numpy as np
from codac import *
from codac.unsupported import *
import datetime
import cv2
from math import pi,floor,ceil
import matplotlib.pyplot as plt
from collections import deque

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
    res = TubeVector(Interval(x[0].tdomain().lb(),x[0].tdomain().lb()+total_time),dt,2)
    # print(res)
    cmt_shift = x[0].tdomain().diam()
    # res[0] &= x[0][0]
    # res[1] &= x[0][1]
    print("x[0][0] = ",x[0][0])
    print("x[0][0].tdomain() = ",x[0][0].tdomain())
    res[0].set(x[0][0],x[0][0].tdomain())
    res[1].set(x[0][1],x[0][1].tdomain())
    print('ok in')
    for i in np.arange(1,len(x)):
        # print("i =",i)
        x[i].shift_tdomain(cmt_shift)
        # print("end shift")
        res[0].set(x[i][0],x[i][0].tdomain())
        res[1].set(x[i][1],x[i][1].tdomain())
        # res[1] &= x[i][1]
        cmt_shift += x[i].tdomain().diam()

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

def InverseTube(x,tdomain,dt):
    map_x = []
    map_y = []
    res = TubeVector(tdomain,dt,2)
    for i in range(x.nb_slices()):
        # res_i = TubeVector(res[0].slice(i).tdomain(),dt,IntervalVector([x(x[0].slice(x.nb_slices() -1 -i).tdomain())[0],x(x[1].slice(x.nb_slices() -1 -i).tdomain())[1]]))
        res.set(IntervalVector([x(x[0].slice(x.nb_slices() -1 -i).tdomain())[0],x(x[1].slice(x.nb_slices() -1 -i).tdomain())[1]]),res[0].slice(i).tdomain())

    return res

def ContourTraj(x_robot,dx_robot,dt,L):
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
   
    #right contour
    x_right = TrajectoryVector(2)
    v_right = TrajectoryVector(2)
    x_right[0] = x_robot[0] + L*sin_theta
    x_right[1] = x_robot[1] - L*cos_theta
    data_vx = []
    data_vy = []
    t = [x_right[0].tdomain().lb()]
    
    while(t[-1]+dt < x_right[0].tdomain().ub()):
        v0 =  (x_right[0](t[-1] + dt) - x_right[0](t[-1]))/dt
        v1 =  (x_right[1](t[-1] + dt) - x_right[1](t[-1]))/dt
        data_vx.append(v0)
        data_vy.append(v1)
        t.append(t[-1]+dt)
   
    if(len(data_vx) > 1 ):
        data_vx.append(data_vx[-1])
        data_vy.append(data_vy[-1])
    else:
        data_vx.append(0)
        data_vy.append(0)
   
    v_right[0] = Trajectory(t, data_vx)
    v_right[1] = Trajectory(t, data_vy)
   
    # v_right[0] = dx_robot[0] #+ L*cos(x_robot[2].sample(dt))*dx_robot[2].sample(dt)
    # v_right[1] = dx_robot[1] #+ L*sin(x_robot[2].sample(dt))*dx_robot[2].sample(dt)
    v[0] = v_right
   
    #left contour
    x_left = TrajectoryVector(2)
    v_left = TrajectoryVector(2)
    x_left[0] = x_robot[0] - L*sin_theta
    x_left[1] = x_robot[1] + L*cos_theta
    # v_left[0] = dx_robot[0] #- L*cos(x_robot[2].sample(dt))*dx_robot[2].sample(dt)
    # v_left[1] = dx_robot[1] #- L*sin(x_robot[2].sample(dt))*dx_robot[2].sample(dt)
    data_vx = []
    data_vy = []
    t = [x_right[0].tdomain().lb()]
    while(t[-1] + dt< x_left[0].tdomain().ub()):
        v0 =  (x_left[0](t[-1] + dt) - x_left[0](t[-1]))/dt
        v1 =  (x_left[1](t[-1] + dt) - x_left[1](t[-1]))/dt
        data_vx.append(v0)
        data_vy.append(v1)
        t.append(t[-1]+dt)
    if(len(data_vx) > 1 ):
        data_vx.append(data_vx[-1])
        data_vy.append(data_vy[-1])
    else:
        data_vx.append(0)
        data_vy.append(0)
    v_left[0] = Trajectory(t, data_vx)
    v_left[1] = Trajectory(t, data_vy)
    v[2] = InverseTraj(v_left,dt)
    
    #right to left
    alpha0 = (((x_left[0](tdomain.ub()) - x_right[0](tdomain.ub()))/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_right[0](tdomain.ub()))
    alpha1 = (((x_left[1](tdomain.ub()) - x_right[1](tdomain.ub()))/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_right[1](tdomain.ub()))
    d_xrl = TrajectoryVector(t, np.array([np.ones((1, len(t))), np.ones((1, len(t)))]))
    d_xrl[0] *= alpha0
    d_xrl[1] *= alpha1
    
    x_rl = TrajectoryVector(2)
    x_rl = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))
    v[1] = d_xrl
    # elif(dimension == 2):
    #     rl_dict_x = dict()
    #     rl_dict_y = dict()
    #     M=move_motif(np.array([[-d,d,d,-d,-d],[L,L,-L,-L,L]]),x_robot(tdomain.ub())[0],x_robot(tdomain.ub())[1],x_robot(tdomain.ub())[2])
    #     rl_dict_x[tdomain.lb()] = x_right(tdomain.ub())[0]
    #     rl_dict_y[tdomain.lb()] = x_right(tdomain.ub())[1]
    #     rl_dict_x[tdomain.diam()*0.33 + tdomain.lb()] = M[0,2]
    #     rl_dict_y[tdomain.diam()*0.33 + tdomain.lb()] = M[1,2]
    #     rl_dict_x[tdomain.diam()*0.66 + tdomain.lb()] = M[0,1]
    #     rl_dict_y[tdomain.diam()*0.66 + tdomain.lb()] = M[1,1]
    #     rl_dict_x[tdomain.ub()] = x_left(tdomain.ub())[0]
    #     rl_dict_y[tdomain.ub()] = x_left(tdomain.ub())[1]
    #     x_rl = TrajectoryVector(2)
    #     x_rl[0] = Trajectory(rl_dict_x)
    #     x_rl[1] = Trajectory(rl_dict_y)
    #
    #     v[1] = TrajectoryVector(2)
    #     t_1 = np.arange(tdomain.lb(),tdomain.diam()*0.33 + tdomain.lb()+dt,dt)
    #     v_1 = TrajectoryVector(t_1, np.array([np.ones((1, len(t_1))), np.ones((1, len(t_1)))]))
    #     v_1[0] *= v_right(tdomain.ub())[0]
    #     v_1[1] *= v_right(tdomain.ub())[1]
    #     t_2 = np.arange(tdomain.diam()*0.66 + tdomain.lb(),tdomain.ub()+dt,dt)
    #     v_2 = TrajectoryVector(t_2, np.array([np.ones((1, len(t_2))), np.ones((1, len(t_2)))]))
    #     v_2[0] *= v_left(tdomain.ub())[0]
    #     v_2[1] *= v_left(tdomain.ub())[1]
    #     v_2.truncate_tdomain(Interval(tdomain.diam()*0.66 + tdomain.lb(),tdomain.ub()))
    #
    #     map_x = v_1[0].sample(dt).sampled_map()
    #     map_y = v_1[1].sample(dt).sampled_map()
    #     map_x.update(d_xrl.truncate_tdomain(Interval(tdomain.diam()*0.33 + tdomain.lb(),tdomain.diam()*0.66 + tdomain.lb()))[0].sample(dt).sampled_map())
    #     map_y.update(d_xrl.truncate_tdomain(Interval(tdomain.diam()*0.33 + tdomain.lb(),tdomain.diam()*0.66 + tdomain.lb()))[1].sample(dt).sampled_map())
    #     map_x.update(v_2[0].sample(dt).sampled_map())
    #     map_y.update(v_2[1].sample(dt).sampled_map())
    #
    #     v[1][0] = Trajectory(map_x)
    #     v[1][1] = Trajectory(map_y)
    #

    #left to right
    t_values = np.arange(tdomain.lb(),tdomain.ub(),dt)
    alpha0 = (((x_right[0](tdomain.lb()) - x_left[0](tdomain.lb()))/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_left[0](tdomain.lb()))
    alpha1 = (((x_right[1](tdomain.lb()) - x_left[1](tdomain.lb()))/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_left[1](tdomain.lb()))
    d_xlr = TrajectoryVector(t_values, np.array([np.ones((1, len(t_values))), np.ones((1, len(t_values)))]))
    d_xlr[0] *= alpha0
    d_xlr[1] *= alpha1
    x_lr = TrajectoryVector(2)
    # if(dimension == 1):
    x_lr = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))
    v[3] = d_xlr
    # elif(dimension == 2):
    #     lr_dict_x = dict()
    #     lr_dict_y = dict()
    #     M=move_motif(np.array([[-d,d,d,-d,-d],[L,L,-L,-L,L]]),x_robot(tdomain.lb())[0],x_robot(tdomain.lb())[1],x_robot(tdomain.lb())[2])
    #     lr_dict_x[tdomain.lb()] = x_left(tdomain.lb())[0]
    #     lr_dict_y[tdomain.lb()] = x_left(tdomain.lb())[1]
    #     lr_dict_x[tdomain.diam()*0.33 + tdomain.lb()] = M[0,0]
    #     lr_dict_y[tdomain.diam()*0.33 + tdomain.lb()] = M[1,0]
    #     lr_dict_x[tdomain.diam()*0.66 + tdomain.lb()] = M[0,3]
    #     lr_dict_y[tdomain.diam()*0.66 + tdomain.lb()] = M[1,3]
    #     lr_dict_x[tdomain.ub()] = x_right(tdomain.lb())[0]
    #     lr_dict_y[tdomain.ub()] = x_right(tdomain.lb())[1]
    #     x_lr = TrajectoryVector(2)
    #     x_lr[0] = Trajectory(lr_dict_x)
    #     x_lr[1] = Trajectory(lr_dict_y)
    #
    #     v[3] = TrajectoryVector(2)
    #     t_1 = np.arange(tdomain.lb(),tdomain.diam()*0.33 + tdomain.lb()+dt,dt)
    #     v_1 = TrajectoryVector(t_1, np.array([np.ones((1, len(t_1))), np.ones((1, len(t_1)))]))
    #     v_1[0] *= -v_left(tdomain.lb())[0]
    #     v_1[1] *= -v_left(tdomain.lb())[1]
    #     t_2 = np.arange(tdomain.diam()*0.66 + tdomain.lb(),tdomain.ub()+dt,dt)
    #     v_2 = TrajectoryVector(t_2, np.array([np.ones((1, len(t_2))), np.ones((1, len(t_2)))]))
    #     v_2[0] *= v_right(tdomain.lb())[0]
    #     v_2[1] *= v_right(tdomain.lb())[1]
    #     v_2.truncate_tdomain(Interval(tdomain.diam()*0.66 + tdomain.lb(),tdomain.ub()))
    #
    #     map_x = v_1[0].sample(dt).sampled_map()
    #     map_y = v_1[1].sample(dt).sampled_map()
    #     map_x.update(d_xlr.truncate_tdomain(Interval(tdomain.diam()*0.33 + tdomain.lb(),tdomain.diam()*0.66 + tdomain.lb()))[0].sample(dt).sampled_map())
    #     map_y.update(d_xlr.truncate_tdomain(Interval(tdomain.diam()*0.33 + tdomain.lb(),tdomain.diam()*0.66 + tdomain.lb()))[1].sample(dt).sampled_map())
    #     map_x.update(v_2[0].sample(dt).sampled_map())
    #     map_y.update(v_2[1].sample(dt).sampled_map())
    #
    #     v[3][0] = Trajectory(map_x)
    #     v[3][1] = Trajectory(map_y)

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

def ContourTube(x_robot,dx_robot,dt,dims,dimension):
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
    L = dims[dimension - 1]
    d = dims[0]

    u_real = TubeVector(tdomain, dt, 2)
    u_real[0] = dx_robot[0]/sqrt((dx_robot[0]*dx_robot[0]) + (dx_robot[1]*dx_robot[1]))
    u_real[1] = dx_robot[1]/sqrt((dx_robot[1]*dx_robot[1]) + (dx_robot[1]*dx_robot[1]))
    print("dx_robot= ",dx_robot)
    #right
    x_right = TubeVector(tdomain,dt,2)
    x_right[0] = x_robot[0]+ L*u_real[1]
    x_right[1] = x_robot[1] - L*u_real[0]

    #left
    x_left = TubeVector(tdomain,dt,2)
    x_left[0] = x_robot[0] - L*u_real[1]
    x_left[1] = x_robot[1] + L*u_real[0]

    #right
    # x_right = TubeVector(tdomain,dt,2)
    v_right = TubeVector(tdomain,dt,2)
    # x_right[0] = x_robot[0] + L*sin(x_robot[2])
    # x_right[1] = x_robot[1] - L*cos(x_robot[2])
    v_right[0] = dx_robot[0]
    v_right[1] = dx_robot[1]
    # v[0] = v_right

    #left
    # x_left = TubeVector(tdomain,dt,2)
    v_left = TubeVector(tdomain,dt,2)
    # x_left[0] = x_robot[0] - L*sin(x_robot[2])
    # x_left[1] = x_robot[1] + L*cos(x_robot[2])
    v_left[0] = -dx_robot[0]
    v_left[1] = -dx_robot[1]
    # v[2] = InverseTube(v_left,dt)

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

    # x_lr = TrajectoryVector(2)
    # x_lr = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    x_rl = TubeVector(x_rl_lb,x_rl_ub,dt)
    x_lr = TubeVector(x_lr_lb,x_lr_ub,dt)


    return ConcatenateTubes([x_right,x_rl,InverseTube(x_left,tdomain,dt),x_lr],dt),[v_right,d_xrl,InverseTube(v_left,tdomain,dt),d_xlr]
    # return ConcatenateTubes([x_right],dt),[v_right,d_xrl,InverseTube(v_left,dt),d_xlr]

def move_motif(M,x,y,th):
    M1=np.ones((1,len(M[1,:])))
    M2=np.vstack((M, M1))
    R = np.array([[np.cos(th),-np.sin(th),x], [np.sin(th),np.cos(th),y]])
    return(R @ M2)

def pointsRectangle(tdomain,dt,x1_robot,x2_robot,d,L):
    x_truth =  TrajectoryVector(tdomain, TFunction("("+x1_robot[0]+";"+x2_robot[0]+")"));
    cos_theta = Trajectory(tdomain, TFunction("cos(atan2("+x2_robot[1]+","+x1_robot[1]+"))"))
    sin_theta = Trajectory(tdomain, TFunction("sin(atan2("+x2_robot[1]+","+x1_robot[1]+"))"))
    dtheta = Trajectory(tdomain, TFunction("("+x2_robot[2]+"*"+x1_robot[1]+"-"+x1_robot[2]+"*"+x2_robot[1]+")/("+x1_robot[1]+"^2 + "+x2_robot[1]+"^2)"))

    plr = TrajectoryVector(2)
    dplr = TrajectoryVector(2)
    plr[0] = x_truth[0] - d*sin_theta.sample(dt) -L*cos_theta.sample(dt)
    plr[1] = x_truth[1] + d*cos_theta.sample(dt) -L*sin_theta.sample(dt)
    dplr[0] = Trajectory(tdomain,TFunction(x1_robot[1])) - d*cos_theta.sample(dt)*dtheta.sample(dt) + L*sin_theta.sample(dt)*dtheta.sample(dt)
    dplr[1] = Trajectory(tdomain,TFunction(x2_robot[1])) - d*sin_theta.sample(dt)*dtheta.sample(dt) - L*cos_theta.sample(dt)*dtheta.sample(dt)
    plf = TrajectoryVector(2)
    dplf = TrajectoryVector(2)
    plf[0] = x_truth[0] - d*sin_theta.sample(dt) +L*cos_theta.sample(dt)
    plf[1] = x_truth[1] + d*cos_theta.sample(dt) +L*sin_theta.sample(dt)
    dplf[0] = Trajectory(tdomain,TFunction(x1_robot[1])) - d*cos_theta.sample(dt)*dtheta.sample(dt) - L*sin_theta.sample(dt)*dtheta.sample(dt)
    dplf[1] = Trajectory(tdomain,TFunction(x2_robot[1])) - d*sin_theta.sample(dt)*dtheta.sample(dt) + L*cos_theta.sample(dt)*dtheta.sample(dt)
    prr = TrajectoryVector(2)
    dprr = TrajectoryVector(2)
    prr[0] = x_truth[0] + d*sin_theta.sample(dt) -L*cos_theta.sample(dt)
    prr[1] = x_truth[1] - d*cos_theta.sample(dt) -L*sin_theta.sample(dt)
    dprr[0] = Trajectory(tdomain,TFunction(x1_robot[1])) + d*cos_theta.sample(dt)*dtheta.sample(dt) + L*sin_theta.sample(dt)*dtheta.sample(dt)
    dprr[1] = Trajectory(tdomain,TFunction(x2_robot[1])) + d*sin_theta.sample(dt)*dtheta.sample(dt) - L*cos_theta.sample(dt)*dtheta.sample(dt)
    prf = TrajectoryVector(2)
    dprf = TrajectoryVector(2)
    prf[0] = x_truth[0] + d*sin_theta.sample(dt) +L*cos_theta.sample(dt)
    prf[1] = x_truth[1] - d*cos_theta.sample(dt) +L*sin_theta.sample(dt)
    dprf[0] = Trajectory(tdomain,TFunction(x1_robot[1])) + d*cos_theta.sample(dt)*dtheta.sample(dt) - L*sin_theta.sample(dt)*dtheta.sample(dt)
    dprf[1] = Trajectory(tdomain,TFunction(x2_robot[1])) + d*sin_theta.sample(dt)*dtheta.sample(dt) + L*cos_theta.sample(dt)*dtheta.sample(dt)

    return plr,dplr,plf,dplf,prr,dprr,prf,dprf,cos_theta,sin_theta,dtheta

def ContourPoint(tdomain,dt,pl,dpl,pr,dpr,d,L):
    x_left = TrajectoryVector(pl)
    x_right = TrajectoryVector(pr)
    #right to left
    #1d
    alpha0 = (((x_left[0](tdomain.ub()) - x_right[0](tdomain.ub()))/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_right[0](tdomain.ub()))
    alpha1 = (((x_left[1](tdomain.ub()) - x_right[1](tdomain.ub()))/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_right[1](tdomain.ub()))
    x_rl = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))
    dx_rl = IntervalVector([Interval(alpha0),Interval(alpha1)])
    #left to right
    #1d
    alpha0 = (((x_right[0](tdomain.lb()) - x_left[0](tdomain.lb()))/(tdomain.ub() - tdomain.lb())))
    beta0 = (x_left[0](tdomain.lb()))
    alpha1 = (((x_right[1](tdomain.lb()) - x_left[1](tdomain.lb()))/(tdomain.ub() - tdomain.lb())))
    beta1 = (x_left[1](tdomain.lb()))
    x_lr = TrajectoryVector(tdomain, TFunction("((t - "+str(tdomain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(tdomain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))
    dx_lr = IntervalVector([Interval(alpha0),Interval(alpha1)])

    v_left_inv = InverseTraj(dpl,dt) #inverse left
    x_left_inv = InverseTraj(x_left,dt)

    #list with the first derivative of each part of the sonar's contours
    v = []
    v.append(dpr)
    v.append(dx_rl)
    v.append(v_left_inv)
    v.append(dx_lr)

    return ConcatenateTraj([x_right,x_rl,x_left_inv,x_lr],dt),v

def CutDomain(domain,domains_to_remove):
    ## domain is an interval
    ## domains_to_remove is a list of intervals
    ## domains_to_remove[i] is a subset or equal to domain
    ## it returns a list of intervals resulting of domain\domains_to_remove
    res =  []
    res.append(domain)
    for d_rem in domains_to_remove:
        aux_res = []
        for i in range(len(res)):
            d = res[i]
            if(not (d&d_rem).is_empty()):
                a,b,c,d = d.lb(),d.ub(),d_rem.lb(),d_rem.ub()
                if(a < c):
                    aux_res.append(Interval(a,c))
                if(d < b):
                    aux_res.append(Interval(d,b))
            else:
                aux_res.append(d)
        res = aux_res.copy()
    return res

def SortLoops(v_list,tdomain):
    l_first = Loop(Interval(tdomain.lb()),Interval(tdomain.ub()))
    sorted_v = dict()
    for i in range(len(v_list)):
        v = v_list[i]
        new_l = Loop(v[0],v[1])

        for l in l_first.subset:
            if(not (new_l.l & l.l).is_empty()):
                if(new_l.l.is_subset(l.l)):
                    l.subset.append(new_l)
                    new_l.master.append(l)
                elif(l.l.is_subset(new_l.l)):
                    new_l.subset.append(l)
                    l.master.append(new_l)
                else:
                    new_l.inter_list.append(l)
                    l.inter_list.append(new_l)
        l_first.subset.append(new_l)
        new_l.master.append(l_first)

    singleton = []


    for l in l_first.subset:
        if not l.nb_inter() in sorted_v.keys():
            sorted_v[l.nb_inter()] = []
        sorted_v[l.nb_inter()].append(l)

    v_key = list(sorted_v.keys())
    v_key.sort()
    list_2 = []
    for i in v_key:
        for v in sorted_v[i]:
            list_2.append(v)

    v_list = []
    while(len(list_2) > 0):
        l = list_2.pop(0)
        if(l.is_single() and l.is_leaf() and len(l.subset_active) == 0):
            l.isolated = True
            v_list.insert(0,l)
            for l_m in l.master:
                l_m.subset.remove(l)
                l_m.subset_active.append(l)
            continue
        if(l.nb_subset() == 0):
            l.isolated = True
            v_list.append(l)
            for l_m in l.master:
                l_m.subset.remove(l)
                l_m.subset_active.append(l)
            for l_inter in l.inter_list:
                if (l_inter in list_2):
                    list_2.remove(l_inter)
                    for l_m in l_inter.master:
                        l_m.subset.remove(l_inter)

        else:
            list_2.append(l)

    return v_list

def cut_angle(ang):
    res = (pi + ang)
    scl_max = res.ub()
    scl_min = res.lb()
    if(scl_max > pi):
        scl_max = -pi + (scl_max - pi)
    elif(scl_max < -pi):
        scl_max = pi + (scl_max + pi)
    if(scl_min > pi):
        scl_min = -pi + (scl_min - pi)
    elif(scl_min < -pi):
        scl_min = pi + (scl_min + pi)
    if(scl_min < scl_max):
        return Interval(scl_min,scl_max)
    return Interval(scl_max,scl_min)

def CreateSep(domain,X,gamma,dt,eps,fig):
    pixel_x = eps
    pixel_y = -eps
    npx = int((X[0].ub() - X[0].lb())/abs(pixel_x))
    npy = int((X[1].ub() - X[1].lb())/abs(pixel_y))
    img_aux = np.zeros((npx, npy), dtype=np.int64)

    t = domain.lb()
    last_t = domain.lb()

    #right to left
    alpha0 = (((gamma[2][0](domain.ub()) - gamma[0][0](domain.ub()))/(domain.ub() - domain.lb())))
    beta0 = (gamma[0][0](domain.ub()))
    alpha1 = (((gamma[2][1](domain.ub()) - gamma[0][1](domain.ub()))/(domain.ub() - domain.lb())))
    beta1 = (gamma[0][1](domain.ub()))
    x_rl = TrajectoryVector(domain, TFunction("((t - "+str(domain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(domain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))
    #left to right
    alpha0 = (((gamma[0][0](domain.lb()) - gamma[2][0](domain.lb()))/(domain.ub() - domain.lb())))
    beta0 = (gamma[2][0](domain.lb()))
    alpha1 = (((gamma[0][1](domain.lb()) - gamma[2][1](domain.lb()))/(domain.ub() - domain.lb())))
    beta1 = (gamma[2][1](domain.lb()))
    x_lr = TrajectoryVector(domain, TFunction("((t - "+str(domain.lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(domain.lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    x = [gamma[0],gamma[2],x_rl,x_lr]
    while(t < domain.ub()):
        t += dt
        if(t > domain.ub()):
            t = domain.ub()

        for i in range(4):
            xi = x[i](Interval(last_t,t))[0]
            yi = x[i](Interval(last_t,t))[1]
            fig.draw_box(IntervalVector([xi,yi]),"[k]")

            x_pix = (xi - X[0].lb())/pixel_x
            y_pix= (yi - X[1].ub())/pixel_y

            for i in range(floor(x_pix.lb()),ceil(x_pix.ub())):
                for j in range(floor(y_pix.lb()),ceil(y_pix.ub())):
                    img_aux[i,j] = 1
                    # img_test[i,j] = 1

        last_t = t

    # plt.imshow(img_test)
    # plt.show()
    contours, hyera = cv2.findContours(np.ascontiguousarray(img_aux.copy(), dtype=np.uint8).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    images_out = []
    for idx in np.arange(0,len(contours),1):
        c = contours[idx]
        if(hyera[0][idx][3] == -1):
            img_out_i = np.ascontiguousarray(img_aux.copy(), dtype=np.uint8)
            cv2.drawContours(img_out_i, [c], contourIdx=0, color=(255,255,255),thickness=-1)
            images_out.append(img_out_i)

    img_out = np.ascontiguousarray(np.zeros((npx, npy), dtype=np.int64), dtype=np.uint8)
    for img in images_out:
        img[img > 0] = 1
        img_out += img

    img_out[(img_out%2) == 0] = 0
    img_in = np.ones((npx, npy), dtype=np.uint8) - img_out
    img_out[img_aux > 0 ] = 1
    img_in[img_aux > 0 ] = 1
    # plt.imshow(img_out)
    # plt.show()
    # plt.imshow(img_in)
    # plt.show()
    img_out = img_out.cumsum(0).cumsum(1)
    img_in = img_in.cumsum(0).cumsum(1)
    ctcOut = CtcRaster(img_out, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
    ctcIn = CtcRaster(img_in, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
    sep = SepCtcPair(ctcIn, ctcOut)
    return sep

def CreateCtc(domain,X,gamma,dt,eps,t_ints,fig):
    pixel_x = eps
    pixel_y = -eps
    npx = int((X[0].ub() - X[0].lb())/abs(pixel_x))
    npy = int((X[1].ub() - X[1].lb())/abs(pixel_y))
    img_aux = np.zeros((npx, npy), dtype=np.int64)

    t = domain.lb()
    last_t = domain.lb()

    #right to left
    alpha0 = (((gamma[2][0](t_ints[0].ub()) - gamma[0][0](t_ints[0].ub()))/(t_ints[0].ub() - t_ints[0].lb())))
    beta0 = (gamma[0][0](t_ints[0].ub()))
    alpha1 = (((gamma[2][1](t_ints[0].ub()) - gamma[0][1](t_ints[0].ub()))/(t_ints[0].ub() - t_ints[0].lb())))
    beta1 = (gamma[0][1](t_ints[0].ub()))
    x_rl = TrajectoryVector(t_ints[0], TFunction("((t - "+str(t_ints[0].lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(t_ints[0].lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))
    #left to right
    alpha0 = (((gamma[0][0](t_ints[1].lb()) - gamma[2][0](t_ints[1].lb()))/(t_ints[1].ub() - t_ints[1].lb())))
    beta0 = (gamma[2][0](t_ints[1].lb()))
    alpha1 = (((gamma[0][1](t_ints[1].lb()) - gamma[2][1](t_ints[1].lb()))/(t_ints[1].ub() - t_ints[1].lb())))
    beta1 = (gamma[2][1](t_ints[1].lb()))
    x_lr = TrajectoryVector(t_ints[1], TFunction("((t - "+str(t_ints[1].lb())+")*"+str(alpha0)+"+"+str(beta0)+";(t - "+str(t_ints[1].lb())+")*"+str(alpha1)+"+"+str(beta1)+")"))

    x = [gamma[0],gamma[2],x_rl,x_lr]
    while(t < domain.ub()):
        t += dt
        if(t > domain.ub()):
            t = domain.ub()

        for i in range(2):
            xi = x[i](Interval(last_t,t))[0]
            yi = x[i](Interval(last_t,t))[1]
            fig.draw_box(IntervalVector([xi,yi]),"[red]")

            x_pix = (xi - X[0].lb())/pixel_x
            y_pix= (yi - X[1].ub())/pixel_y
            for i in range(floor(x_pix.lb()),ceil(x_pix.ub())):
                for j in range(floor(y_pix.lb()),ceil(y_pix.ub())):
                    img_aux[i,j] = 1

        last_t = t

    t = t_ints[0].lb()
    last_t = t_ints[0].lb()
    while(t < t_ints[0].ub()):
        t += dt
        if(t > t_ints[0].ub()):
            t = t_ints[0].ub()

        xi = x[2](Interval(last_t,t))[0]
        yi = x[2](Interval(last_t,t))[1]
        fig.draw_box(IntervalVector([xi,yi]),"[red]")
        x_pix = (xi - X[0].lb())/pixel_x
        y_pix= (yi - X[1].ub())/pixel_y
        for i in range(floor(x_pix.lb()),ceil(x_pix.ub())):
            for j in range(floor(y_pix.lb()),ceil(y_pix.ub())):
                img_aux[i,j] = 1

        last_t = t

    t = t_ints[1].lb()
    last_t = t_ints[1].lb()
    while(t < t_ints[1].ub()):
        t += dt
        if(t > t_ints[1].ub()):
            t = t_ints[1].ub()

        xi = x[3](Interval(last_t,t))[0]
        yi = x[3](Interval(last_t,t))[1]
        fig.draw_box(IntervalVector([xi,yi]),"[red]")
        x_pix = (xi - X[0].lb())/pixel_x
        y_pix= (yi - X[1].ub())/pixel_y

        for i in range(floor(x_pix.lb()),ceil(x_pix.ub())):
            for j in range(floor(y_pix.lb()),ceil(y_pix.ub())):
                img_aux[i,j] = 1
                # img_test[i,j] = 1

        last_t = t

    # plt.imshow(img_test)
    # plt.show()
    contours, hyera = cv2.findContours(np.ascontiguousarray(img_aux.copy(), dtype=np.uint8).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    images_out = []
    for idx in np.arange(0,len(contours),1):
        c = contours[idx]
        if(hyera[0][idx][3] == -1):
            img_out_i = np.ascontiguousarray(img_aux.copy(), dtype=np.uint8)
            cv2.drawContours(img_out_i, [c], contourIdx=0, color=(255,255,255),thickness=-1)
            images_out.append(img_out_i)

    img_out = np.ascontiguousarray(np.zeros((npx, npy), dtype=np.int64), dtype=np.uint8)
    for img in images_out:
        img[img > 0] = 1
        img_out += img

    img_out[(img_out%2) == 0] = 0
    img_in = np.ones((npx, npy), dtype=np.uint8) - img_out
    img_out[img_aux > 0 ] = 1
    img_in[img_aux > 0 ] = 1
    # plt.imshow(img_out)
    # plt.show()
    # plt.imshow(img_in)
    # plt.show()
    img_out = img_out.cumsum(0).cumsum(1)
    img_in = img_in.cumsum(0).cumsum(1)
    ctcOut = CtcRaster(img_out, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
    ctcIn = CtcRaster(img_in, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
    sep = SepCtcPair(ctcIn, ctcOut)
    # plt.imshow(img_out)
    # plt.show()
    # img_out = img_out.cumsum(0).cumsum(1)
    # ctcOut = CtcRaster(img_out, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
    return sep

def CreateAllSeps(X,cuts,gamma,tdomain,dt,eps):
    # pixel_x = eps
    # pixel_y = -eps
    # npx = int((X[0].ub() - X[0].lb())/abs(pixel_x))
    # npy = int((X[1].ub() - X[1].lb())/abs(pixel_y))
    # img_test = np.zeros((npx, npy), dtype=np.int64)

    fig = VIBesFigMap("Sivia1")
    fig.set_properties(100, 100, 800, 800)

    seps = []
    cut_seps = []
    # print("len(cuts) = ",len(cuts))
    for i in range(len(cuts)):
        if(i == 0):
            int_t  = Interval(tdomain.lb(),cuts[i].ub())
        else:
            int_t  = Interval(cuts[i-1].lb(),cuts[i].ub())

        if(len(cuts) > i+1):
            next_int_t = Interval(cuts[i].lb(),cuts[i+1].ub())
        else:
            next_int_t = Interval(cuts[i].lb(),tdomain.ub())
        t_ints = [int_t,next_int_t]
        seps.append(CreateSep(int_t,X,gamma,dt,eps,fig))
        cut_seps.append(CreateCtc(cuts[i],X,gamma,dt,eps,t_ints,fig))
    int_t  = Interval(cuts[len(cuts) - 1].lb(),tdomain.ub())
    seps.append(CreateSep(int_t,X,gamma,dt,eps,fig))
    X = IntervalVector([[3.423050,3.47817],[5.06093,5.10971]])
    fig.draw_box(X,"green[]")
    return seps,cut_seps

def CutTraj(tdomain,v_rob,cut_angl,dt):
    ts = [tdomain]
    cumul = Interval.EMPTY_SET
    int_cumul = Interval.EMPTY_SET
    cuts = []
    while(len(ts) > 0):
        t = ts.pop(0)
        dxt,dyt= v_rob(t)[0],v_rob(t)[1]
        val_angl = atan2(dyt,dxt)
        if(not (cut_angl&val_angl).is_empty()):
            if(t.diam() > dt):
                ts.insert(0,Interval(t.lb() + t.diam()/2.,t.ub()))
                ts.insert(0,Interval(t.lb(),t.lb() + t.diam()/2.))
            else:
                if(Interval(-pi,pi).is_subset(Interval(val_angl).inflate(0.1))):
                    if(Interval(cut_angl).is_subset(Interval(pi).inflate(0.1)) or Interval(cut_angl).is_subset(Interval(-pi).inflate(0.1))):
                        cuts.append(t.inflate(2*dt))
                        cut_angl = cut_angle(val_angl)
                else:
                    if(int_cumul.is_empty()):
                        cumul = t
                        int_cumul = (cut_angl&val_angl)
                    else:
                        cumul |= t
                        int_cumul |= (cut_angl&val_angl)

                    if(cut_angl.is_subset(val_angl)):
                        cumul = t
                        int_cumul = (cut_angl&val_angl)

                    if(cut_angl.is_subset(int_cumul) or cut_angl == int_cumul):
                        cuts.append(cumul.inflate(2*dt))
                        cut_angl = cut_angle(val_angl)
                        cumul = Interval.EMPTY_SET
                        int_cumul = Interval.EMPTY_SET
                        ts = [Interval(t.lb(),tdomain.ub())]
    return cuts


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