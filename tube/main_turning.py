import numpy as np
from math import pi
from codac import *
from vibes import vibes
import datetime
from collections import deque
from winding_lib import *
from topo_lib import *

Tube.enable_syntheses() # faster integral computations

total_time_begin = datetime.datetime.now()

#comment the the examples that should not been used below:

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
dt=0.01
#Range of visibility on each side
L = 5.0
#Area to classify
world = IntervalVector([[-20,40],[-25,40]]) 
#size of the robot for visualization
robot_size = 2.

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

######
#create tubes from equations (parametric equations can be replaced by real data 

# true unknown values
# x_truth is the robot's pose (position and orientation)
# dx_robot its velocity
# ddx_robot its acceleration
x_truth =  TrajectoryVector(tdomain, TFunction("("+x1_robot+";"+x2_robot+")"))
dx_robot =  TrajectoryVector(tdomain, TFunction("("+dx1_robot+";"+dx2_robot+")"))
ddx_robot =  TrajectoryVector(tdomain, TFunction("("+ddx1_robot+";"+ddx2_robot+")"))

#add incertitude
a_robot = TubeVector(ddx_robot,dt) #uncertain robot's acceleration
v_robot = TubeVector(dx_robot,dt) #uncertain robot's speed
v_robot.inflate(0.001) #add incertitude to v_robot

x = TubeVector(tdomain,dt,2) #uncertain robot's position
x0 = x_truth(tdomain.lb()) #initial pose is known
x.set(x0, tdomain.lb())
ctc.deriv.contract(x, v_robot)

######
#create the sensor's contour gamma

#v is a vector with the speed on each of the four parts that are concatenated to create the sensor's contour
x_right,x_rl,x_left,x_lr,gamma,v = ContourTube(x,v_robot,a_robot,dt,L) 
x_left = InverseTube(x_left)
######
#Graphics with Vibes 

beginDrawing()
fig_map = VIBesFigMap("Map")
fig_map.smooth_tube_drawing(True)
fig_map.set_tube_max_disp_slices(10000)
fig_map.set_properties(100, 100, 800, 800)
fig_map.axis_limits(-13,13,-16,9)
# fig_map.add_trajectory(x_truth, "x", 0, 1,"red")
# fig_map.add_tube(x, "[x]", 0, 1)


fig_map.add_tube(x_right, "[x_right]", 0, 1)
# fig_map.add_tube(x_right_plus, "[x_right_plus]", 0, 1)
# fig_map.set_tube_color(x_right_plus,"darkGray[darkGray]")

fig_map.add_tube(x_left, "[x_left]", 0, 1)
# fig_map.add_tube(x_left_plus, "[x_left_plus]", 0, 1)
# fig_map.set_tube_color(x_left_plus,"darkGray[darkGray]")

fig_map.add_tube(x_rl, "[x_rl]", 0, 1)
# fig_map.set_tube_color(x_rl,"darkGray[darkGray]")
fig_map.add_tube(x_lr, "[x_lr]", 0, 1)
# fig_map.set_tube_color(x_lr,"darkGray[darkGray]")

# fig_map.add_tube(gamma, "[gamma]", 0, 1)
# fig_map.set_tube_color(gamma_plus,"darkGray[darkGray]")

# for i in range(len(list_gamma_minus)):
#     fig_map.add_tube(list_gamma_minus[i], "[list_gamma_minus]_"+str(i), 0, 1)
#     fig_map.set_tube_color(list_gamma_minus[i],"darkGray[darkGray]")

# fig_map.add_tube(x_left, "[x_left]", 0, 1)
# fig_map.add_tube(x_rl, "[x_rl]", 0, 1)
# fig_map.add_tube(x_lr, "[x_lr]", 0, 1)
# fig_map.draw_vehicle([x_truth[0](tdomain.ub()),x_truth[1](tdomain.ub()),atan2(dx_robot[1](tdomain.ub()),dx_robot[0](tdomain.ub()))], 2)
# fig_map.draw_box(x(tdomain.ub()),"red[]")
# fig_map.draw_box(x_rl(tdomain.lb()),"k[]")
# fig_map.draw_box(x_rl(tdomain.ub()),"k[]")
# fig_map.add_tube(x_lr, "[x_lr]", 0, 1)
# fig_map.set_tube_color(x_right,"k[k]")
# fig_map.set_tube_color(x_left,"red[red]")
# fig_map.add_tube(gamma, "[gamma]", 0, 1)
# for l in loops:
#     fig_map.draw_box(gamma_plus(l[0]),"k[]")
#     fig_map.draw_box(gamma_plus(l[1]),"k[]")
# fig_map.draw_vehicle([x_truth[0](tdomain.ub()),x_truth[1](tdomain.ub()),atan2(dx_robot[1](tdomain.ub()),dx_robot[0](tdomain.ub()))], robot_size)
fig_map.show()

x_list = [x_right,x_rl,x_left,x_lr]

cm = Interval(0)
D =  IntervalVector([[0,0],[0,oo]]) 
zero_int = IntervalVector([[0,0],[0,0]]) 
for x in x_list:
    t_s = x.tdomain()
    print('t_s = ',t_s)
    t_boxes = [t_s]

    while (len(t_boxes) > 0 ):
        box = x(t_boxes.pop(0))

        if((D & box).is_empty()): 
            last_non_inter = box
        elif(not (D & box).is_empty() and not zero_int.is_subset(box)): 
            if(last_box[0].lb() > 0):
                cm += Interval(0,1)
            elif(last_box[0].ub() < 0):
                cm += Interval(-1,0)
        # elif(zero_int.is_subset(box) ): 
        #     if(last_box[0].lb() > 0):
        #         cm += Interval(0,1)
        #     elif(last_box[0].ub() < 0):
        #         cm += Interval(-1,0)

        

        last_box = box
    # if (0 ) 





