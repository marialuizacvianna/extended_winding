import numpy as np
from math import pi
from codac import *
from vibes import vibes
import datetime
from collections import deque
from winding_lib import *
from topo_lib import *

total_time_begin = datetime.datetime.now()

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

##################### create trajectory from equations (parametric equations can be replaced by real data) #####################
# x_truth is the robot's pose (position and orientation)
# dx_robot its velocity
# ddx_robot its acceleration
x_truth =  TrajectoryVector(tdomain, TFunction("("+x1_robot+";"+x2_robot+";atan2("+dx2_robot+","+dx1_robot+"))"))
dx_robot =  TrajectoryVector(tdomain, TFunction("("+dx1_robot+";"+dx2_robot+";("+ddx2_robot+"*"+dx1_robot+"-"+ddx1_robot+"*"+dx2_robot+")/("+dx1_robot+"^2 + "+dx2_robot+"^2))"))
ddx_robot =  TrajectoryVector(tdomain, TFunction("("+ddx1_robot+";"+ddx2_robot+")"))

##################

grid = []
eps = 1
stack = deque([IntervalVector(world)])
lf = LargestFirst(eps/2.0)

lines = world[1].ub() - world[1].lb() 
cols = world[0].ub() - world[0].lb() 
line = -1
beginDrawing()
fig_siv = VIBesFigMap("SIVIA")
fig_siv.set_properties(100, 100, 800, 800)
fig_siv.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
stack = deque([IntervalVector(world)])

for i in range(int(lines)):
     line += 1
     int_x = Interval(world[1].ub() - line - eps,world[1].ub() - line)
     col = -1
     for j in range(int(cols)):
          col += 1
          int_y = Interval(world[0].lb() + col,world[0].lb() + col + 1)
          X = IntervalVector([int_y,int_x]) 
          grid.append(X)
          fig_siv.draw_box(X,"gray[w]")
# while len(stack) > 0:
#     X = stack.popleft()
#     if(X.max_diam() > eps):
#         (X1, X2) = lf.bisect(X)
#         stack.append(X1)
#         stack.append(X2)
#     else:
#         grid.append(X)


# for block in grid:
#         fig_siv.draw_box(block,"gray[w]")

for t in np.arange(tdomain.lb(),tdomain.ub(),dt):
    if(t > tdomain.ub()):
        t = tdomain.ub()
    x = (x_truth(t)[0])
    y = (x_truth(t)[1])
    theta = (x_truth(t)[2])
    F_x = Function("x", "y", "(cos(%f)*(x - %f) + sin(%f)*(y - %f))^2 -4"%(theta,x,theta,y))
    F_y = Function("x", "y", "(-sin(%f)*(x - %f) + cos(%f)*(y - %f))^2 - %f^2"%(theta,x,theta,y,L))
    sep = SepFwdBwd(F_x,Interval(-oo,0)) &  SepFwdBwd(F_y, Interval(-oo,0))
    for block in grid:
        x_in, x_out = block.copy(),block.copy()
        sep.separate(x_in, x_out)
        if(not (x_out[0].is_empty() or x_out[1].is_empty())):
            fig_siv.draw_box(block,"gray[green]")

endDrawing()




total_time_end = datetime.datetime.now()

print('total time = '+str((total_time_end - total_time_begin).total_seconds())+"s")