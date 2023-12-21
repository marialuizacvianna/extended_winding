import numpy as np
from math import pi
from codac import *
from vibes import vibes
import datetime
from collections import deque
from winding_lib import *
from new_topo_lib import *
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString


total_time_begin = datetime.datetime.now()
dimension = 1
#comment the the examples that should not been used below

##################### Example from the Article #####################
# Equations for creating trajectory
x1_robot = "(-9.6* t^2  + 11.4*(0.5* t^3 - t) +30)"
dx1_robot = "(-19.2* t + 17.1* t^2 - 11.4)"
ddx1_robot = "(-19.2 + 2*17.1* t)"
x2_robot = "(-9.6*(0.5* t^3 - t) - 11.4* t^2 + 30)"
dx2_robot = "(-14.4* t^2 +9.6 -22.8* t)"
ddx2_robot = "(-14.4*2* t -22.8)"
#mission time interval
tdomain = Interval(-1.8,1.8) 
#time step
dt=0.1
#Range of visibility on each side
L = 5.0
#Area to classify
world = IntervalVector([[-20,40],[-25,40]]) 
#size of the robot for visualization
robot_size = 2.

##################### Example no self-inter #####################
# Equations for creating trajectory
x1_robot = "t"
dx1_robot = "1"
ddx1_robot = "0"
x2_robot = "10"
dx2_robot = "0"
ddx2_robot = "0"
#mission time interval
tdomain = Interval(0,15) 
#time step
dt=0.1
#Range of visibility on each side
L = 5.0
#Area to classify
world = IntervalVector([[-10,25],[0,25]]) 
#size of the robot for visualization
robot_size = 2.

##################### Example 2 #####################
# Equations for creating trajectory
x1_robot = "(5* t-5*sin(10* t))"
dx1_robot = "(5-50*cos(10* t))"
ddx1_robot = "(500*sin(10* t))"
x2_robot = "(2-3*cos(10* t))"
dx2_robot = "(30*sin(10* t))"
ddx2_robot = "(300*cos(10* t))"
#mission time interval
tdomain = Interval(-0.02,1.0)
#time step
dt=0.001
#Range of visibility on each side
L = 0.5
#Area to classify
world = IntervalVector([[-6,10],[-5,8]])
#size of the robot for visualization
robot_size = 1.

##################### Example with Sweep Back #####################
# Equations for creating trajectory
x1_robot = "(8*cos( t))"
dx1_robot = "(-8*sin( t))"
ddx1_robot = "(-8*cos( t))"
x2_robot = "(5*sin(2* t) - t)"
dx2_robot = "(10*cos(2* t) - 1)"
ddx2_robot = "(-20*sin(2* t))"
#mission time interval
tdomain = Interval(0,2*pi)
tdomain = Interval(0,0.65 )
#time step
dt=0.01
#Range of visibility on each side
L = 3.6
#Area to classify
world = IntervalVector([[-20,20],[-18,12]])
#size of the robot for visualization
robot_size = 2.

##################### create trajectory from equations (parametric equations can be replaced by real data) #####################
# x_truth is the robot's pose (position and orientation)
# dx_robot its velocity
# ddx_robot its acceleration
x_truth =  TrajectoryVector(tdomain, TFunction("("+x1_robot+";"+x2_robot+";atan2("+dx2_robot+","+dx1_robot+"))"))
dx_robot =  TrajectoryVector(tdomain, TFunction("("+dx1_robot+";"+dx2_robot+";("+ddx2_robot+"*"+dx1_robot+"-"+ddx1_robot+"*"+dx2_robot+")/("+dx1_robot+"^2 + "+dx2_robot+"^2))"))
ddx_robot =  TrajectoryVector(tdomain, TFunction("("+ddx1_robot+";"+ddx2_robot+")"))
#create the sensor's contour gamma
#v is a vector with the speed on each of the four parts that are concatenated to create the sensor's contour
gamma,v = ContourTraj(x_truth,dx_robot,ddx_robot,dt,L,dimension) 

##################### separate gamma into gamma + and gamma - #####################
gamma_plus,v_plus,yt_right,yt_left = GammaPlus(dt,x_truth,dx_robot,ddx_robot,L,dimension)
if(len(yt_right) == 0 and len(yt_left) == 0):
    gamma_plus = TrajectoryVector(gamma)
    v_plus = v


# ##################### Graphics with Vibes #####################
beginDrawing()
fig_map = VIBesFigMap("Map")
fig_map.set_properties(100, 100, 800, 800)
fig_map.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
fig_map.add_trajectory(x_truth, "x", 0, 1,"red")
fig_map.add_trajectory(gamma, "blue", 0, 1)
# fig_map.add_trajectory(gamma_plus, "green", 0, 1)
fig_map.draw_vehicle(tdomain.ub(), x_truth, robot_size)



# ##################### Draw Vectors #####################
d_theta = ((ddx_robot[1]*dx_robot[0]) - (ddx_robot[0]*dx_robot[1]))/(dx_robot[0]*dx_robot[0] + dx_robot[1]*dx_robot[1])

dl = 0.5
l = -L - dl
vectors = []
while(l < L): 
    l += dl
    if(l < -L):
        l = -L
    elif(l > L):
        l = L

    dx = dx_robot[0](tdomain.ub()) + l*cos(x_truth(tdomain.ub())[2])*d_theta(tdomain.ub())
    dy = dx_robot[1](tdomain.ub()) + l*sin(x_truth(tdomain.ub())[2])*d_theta(tdomain.ub())

    dict_x = dict()
    dict_y = dict()
    dict_x[tdomain.lb()] = x_truth[0](tdomain.ub()) + l*sin(x_truth(tdomain.ub())[2])
    dict_y[tdomain.lb()] = x_truth[1](tdomain.ub()) - l*cos(x_truth(tdomain.ub())[2])
    dict_x[tdomain.ub()] = x_truth[0](tdomain.ub()) + l*sin(x_truth(tdomain.ub())[2]) + dx*0.5
    dict_y[tdomain.ub()] = x_truth[1](tdomain.ub()) - l*cos(x_truth(tdomain.ub())[2]) + dy*0.5

    vectors.append(TrajectoryVector(2))
    vectors[-1][0] = Trajectory(dict_x)
    vectors[-1][1] = Trajectory(dict_y)
    fig_map.add_trajectory(vectors[-1], "v_"+str(l), 0, 1,"red")
fig_map.show(0.)
vibes.endDrawing()