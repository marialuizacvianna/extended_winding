import numpy as np
from math import pi
from codac import *
from vibes import vibes
import datetime
from collections import deque
from winding_lib import *
from topo_lib import *

total_time_begin = datetime.datetime.now()
dimension = 1
#comment the the examples that should not been used below

##################### Example from the Article #####################
# Equations for creating trajectory
x1_robot = "(-9.6* (t-1.8)^2  + 11.4*(0.5* (t-1.8)^3 - (t-1.8)) +30)"
dx1_robot = "(-19.2* (t-1.8) + 17.1* (t-1.8)^2 - 11.4)"
ddx1_robot = "(-19.2 + 2*17.1* (t-1.8))"
x2_robot = "(-9.6*(0.5* (t-1.8)^3 - (t-1.8)) - 11.4* (t-1.8)^2 + 30)"
dx2_robot = "(-14.4* (t-1.8)^2 +9.6 -22.8* (t-1.8))"
ddx2_robot = "(-14.4*2* (t-1.8) -22.8)"
#mission time interval
# tdomain = Interval(-1.8,1.8) 
tdomain = Interval(0,3.6) 
#time step
dt=0.01
#Range of visibility on each side
L = 5.0
#Area to classify
world = IntervalVector([[-20,40],[-25,40]]) 
#size of the robot for visualization
robot_size = 2.

##################### Example no self-inter #####################
# Equations for creating trajectory
# x1_robot = "t"
# dx1_robot = "1"
# ddx1_robot = "0"
# x2_robot = "10"
# dx2_robot = "0"
# ddx2_robot = "0"
# #mission time interval
# tdomain = Interval(0,15) 
# #time step
# dt=0.1
# #Range of visibility on each side
# L = 5.0
# #Area to classify
# world = IntervalVector([[-10,25],[0,25]]) 
# #size of the robot for visualization
# robot_size = 2.

##################### Example 2 #####################
# Equations for creating trajectory
# x1_robot = "(5* t-5*sin(10* t))"
# dx1_robot = "(5-50*cos(10* t))"
# ddx1_robot = "(500*sin(10* t))"
# x2_robot = "(2-3*cos(10* t))"
# dx2_robot = "(30*sin(10* t))"
# ddx2_robot = "(300*cos(10* t))"
# #mission time interval
# tdomain = Interval(-0.02,1.0)
# #time step
# dt=0.001
# #Range of visibility on each side
# L = 0.5
# #Area to classify
# world = IntervalVector([[-6,10],[-5,8]])
# #size of the robot for visualization
# robot_size = 1.

##################### Example with Sweep Back #####################
# Equations for creating trajectory
# x1_robot = "(8*cos( t))"
# dx1_robot = "(-8*sin( t))"
# ddx1_robot = "(-8*cos( t))"
# x2_robot = "(5*sin(2* t) - t)"
# dx2_robot = "(10*cos(2* t) - 1)"
# ddx2_robot = "(-20*sin(2* t))"
# #mission time interval
# tdomain = Interval(0,2*pi)
# #time step
# dt=0.01
# #Range of visibility on each side
# L = 3.6
# #Area to classify
# world = IntervalVector([[-20,20],[-18,12]])
# #size of the robot for visualization
# robot_size = 2.

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

##################### find self-intersections in gamma_plus #####################
tplane = TPlane(gamma_plus.tdomain())
tplane.compute_detections(dt, TubeVector(TrajectoryVector(gamma_plus),dt))
tplane.compute_proofs(TubeVector(gamma_plus,dt))
loops = tplane.proven_loops()

##################### derivatives in self-intersections #####################
d_list_i,d_list_f = TangentLoop(v_plus,tdomain,loops)

#####################  create graph and update edges ##################### 
g = Graph(loops,gamma_plus.tdomain(),[d_list_i,d_list_f],yt_right,yt_left)
g.UpdateEdges()

##################### Graphics with Vibes #####################
beginDrawing()
fig_map = VIBesFigMap("Map")
fig_map.set_properties(100, 100, 800, 800)
fig_map.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
fig_map.add_trajectory(x_truth, "x", 0, 1,"red")
fig_map.add_trajectory(gamma, "blue", 0, 1)
fig_map.add_trajectory(gamma_plus, "green", 0, 1)
fig_map.draw_vehicle(tdomain.ub(), x_truth, robot_size)
# for l in loops:
#     fig_map.draw_box(gamma_plus(l[0]),"k[]")
#     fig_map.draw_box(gamma_plus(l[1]),"k[]")
fig_map.show(0.)

##################### Create separators from winding sets #####################
pixel = 0.05 # separators precision in image contractor
seps,back_seps,contour_sep = g.CreateAllSeps(world,gamma,gamma_plus,dt,pixel)

##################### SIVIA #####################
epsilon = 0.05 # sivia's precision

vibes.beginDrawing()
fig_siv = VIBesFigMap("SIVIA")
fig_siv.set_properties(100, 100, 800, 800)
fig_siv.axis_limits(-L-0.5,L+0.5,tdomain.lb()-0.05,tdomain.ub()+0.05)
stack = deque([IntervalVector([[-L,L],[tdomain.lb(),tdomain.ub()]]) ])
res_y, res_in, res_out = [], [], []
lf = LargestFirst(epsilon/2.0)
k = 0
while len(stack) > 0:
    X = stack.popleft()
    box_mosaic_x = x_truth[0](X[1]) + X[0]*sin(x_truth[2](X[1]))
    box_mosaic_y = x_truth[1](X[1]) - X[0]*cos(x_truth[2](X[1]))
    box_mosaic = IntervalVector([box_mosaic_x,box_mosaic_y])
    k = k+1
    nb_in = Interval(0)

    for wn in seps.keys():
        sep = seps[wn]
        x_in, x_out = box_mosaic.copy(),box_mosaic.copy()
        sep.separate(x_in, x_out)
        if(x_in[0].is_empty() or x_in[1].is_empty()):  #it means that box is inside completely
            nb_in += Interval(1)
        elif(not (x_out[0].is_empty() or x_out[1].is_empty())): #partially inside
            nb_in += Interval(0,1)

    if(len(back_seps) > 0):
        for back_sep in back_seps:
            sep = back_sep
            x_in, x_out = box_mosaic.copy(),box_mosaic.copy()
            sep.separate(x_in, x_out)
            if(x_in[0].is_empty() or x_in[1].is_empty()):  #it means that box is inside completely
                nb_in += Interval(1)
            elif(not (x_out[0].is_empty() or x_out[1].is_empty())): #partially inside
                nb_in += Interval(0,1)

    x_in, x_out = box_mosaic.copy(),box_mosaic.copy()
    contour_sep.separate(x_in, x_out)
    if(nb_in.ub() > nb_in.lb()):
        # print("box_mosaic.max_diam() = ",box_mosaic.max_diam())
        if(X.max_diam() <= epsilon or x_in[0].is_empty() or x_in[1].is_empty()):
            if((nb_in) == Interval(0,1)):
                fig_siv.draw_box(X,"gray[magenta]")
            elif((nb_in) == Interval(1,2)):
                fig_siv.draw_box(X,"gray[darkGray]")
            elif((nb_in) == Interval(0,2)):
                fig_siv.draw_box(X,"k[k]")
            # fig_siv.draw_box(X,"k[k]")
        else:
            # print("X = ", X)
            (X1, X2) = lf.bisect(X)
            stack.append(X1)
            stack.append(X2)
    else:
        if((nb_in) == Interval(0)):
            fig_siv.draw_box(X,"gray[w]")
        elif((nb_in) == Interval(1)):
            fig_siv.draw_box(X,"gray[#CCFF99]")
        elif((nb_in) == Interval(2)):
            fig_siv.draw_box(X,"gray[#CCE5FF]")
        elif((nb_in) == Interval(3)):
            fig_siv.draw_box(X,"gray[pink]")
        elif((nb_in) == Interval(4)):
            fig_siv.draw_box(X,"gray[gray]")
        elif((nb_in) == Interval(5)):
            fig_siv.draw_box(X,"gray[orange]")

total_time_end = datetime.datetime.now()

print('total time = '+str((total_time_end - total_time_begin).total_seconds())+"s")