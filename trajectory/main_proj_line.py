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

##
X, Y= [], []
x,y = Interval(0), Interval(0)
seps = []
print("Compute trajectory")
i = 0
for t in np.arange(tdomain.lb(),tdomain.ub(),dt):
  if(t > tdomain.ub()):
    t = tdomain.ub()
  if i%1 == 0:
    x = (x_truth(t)[0])
    y = (x_truth(t)[1])
    theta = (x_truth(t)[2])
    F_x = Function("x", "y", "(cos(%f)*(x - %f) + sin(%f)*(y - %f))^2 -4"%(theta,x,theta,y))
    F_y = Function("x", "y", "(-sin(%f)*(x - %f) + cos(%f)*(y - %f))^2 - %f^2"%(theta,x,theta,y,L))
    seps.append(SepFwdBwd(F_x,Interval(-oo,0)) &  SepFwdBwd(F_y, Interval(-oo,0)))
    X += [Interval(x)]
    Y += [Interval(y)] 
  i += 1

sep = SepUnion(seps)


print("Run SIVIA")
# vibes.newFigure('Path Exploration')
# vibes.setFigureProperties(dict(x=0, y=10, width=500, height=500))
# vibes.drawBox(world[0].lb(), world[0].ub(), world[1].lb(), world[1].ub(), 'b')
# vibes.axisEqual()
#params = {'color_in':'darkGray[w]', 'color_out':'darkGray[k]', 'color_maybe':'lightGray[lightGray]'}
# SIVIA(world, sep, 3)
# beginDrawing()
# x = 0
# y = 0
# theta = (x_truth(tdomain.lb())[2])
# F_x = Function("x", "y", "(cos(%f)*(x - %f) + sin(%f)*(y - %f))^2 "%(theta,x,theta,y))
# F_y = Function("x", "y", "(-sin(%f)*(x - %f) + cos(%f)*(y - %f))^2 - %f^2"%(theta,x,theta,y,L))
# sep = SepFwdBwd(F_x,Interval(-oo,0)) &  SepFwdBwd(F_y, Interval(-oo,0))
# print("x = ", x)
# print("y = ", y)
beginDrawing()
fig = VIBesFig('SIVIA')
fig.set_properties(x=0, y=10, width=500, height=500)
SIVIA(world, sep,0.1)
endDrawing()
# for bx, by in zip(X,Y):
#   vibes.drawBox(bx.lb(), bx.ub(), by.lb(), bybx.ub(), 'k')
# vibes.drawArrow([-400, -50], [-400, 0], 10, 'w[w]')
# vibes.drawArrow([-400, -50], [-350, -50], 10, 'w[w]')
# vibes.axisEqual()

##

total_time_end = datetime.datetime.now()

print('total time = '+str((total_time_end - total_time_begin).total_seconds())+"s")