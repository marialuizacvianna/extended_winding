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
dimension = 2
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

# ##################### Graphics with Vibes #####################
beginDrawing()
fig_map = VIBesFigMap("Map")
fig_map.set_properties(100, 100, 800, 800)
fig_map.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
fig_map.add_trajectory(x_truth, "x", 0, 1,"red")
fig_map.add_trajectory(gamma, "blue", 0, 1)
fig_map.add_trajectory(gamma_plus, "green", 0, 1)
fig_map.draw_vehicle(tdomain.ub(), x_truth, robot_size)
fig_map.show(0.)

# ##################### Create Vertices on Self Intersections #####################

map_x = gamma_plus[0].sample(dt).sampled_map()
map_y = gamma_plus[1].sample(dt).sampled_map()

keys = list(map_x)


val_x = [map_x[x] for x in keys]
val_y = [map_y[x] for x in keys]

points = list(zip(val_x, val_y))

curve = LineString(points)

points_loops = []
idx_loops_before = []
idx_loops_after = []
time_loops = []
V = []
idx_list = []
idx_count = 0
if (not curve.is_simple):
    intersections = list((curve.intersection(curve)).geoms)

    for i in range(len(intersections)):
        intersection = intersections[i]
        
        point_0 = Point(intersection.coords[0][0],intersection.coords[0][1])
        
        if(intersection.coords[0] not in points and point_0.coords() not in points_loops):
            fig_map.draw_circle(point_0.pos_x,point_0.pos_y,0.1,"red[red]")
            new_v = Vertice(point_0.pos_x,point_0.pos_y,idx_count)
            points_loops.append(point_0.coords())
            found_before = -1
            found_after = -1

            if(i-5 > 0):
                found_before = i-5
                idx_before = points.index(intersections[found_before].coords[0])
                point_before_i = (val_x[idx_before],val_y[idx_before])
                new_v.t_before_i = keys[points.index(point_before_i)]

            if(i + 5 < len(intersections)):
                found_after = i+5
                idx_after = points.index(intersections[found_after].coords[1])
                point_after_i = (intersections[found_after].coords[1][0],intersections[found_after].coords[1][1])
                new_v.t_after_i = keys[points.index(point_after_i)]
                dx = (val_x[idx_after] - new_v.point.pos_x)/abs(val_x[idx_after] - new_v.point.pos_x)
                dy = (val_y[idx_after] - new_v.point.pos_y)/abs(val_y[idx_after] - new_v.point.pos_y)
                new_v.di = IntervalVector([[dx,dx],[dy,dy]])

            V.append(new_v)
            idx_list.append(idx_count)
            idx_count += 1

        elif(intersection.coords[0] not in points and point_0.coords() in points_loops):    
            idx_v = -1
            it = 0
            while(idx_v < 0 and it<len(V)):
                if(point_0.coords() == V[it].point.coords()):
                    idx_v = it
                else:
                    it += 1
            
            idx_list.append(idx_v)
            found_before = -1
            found_after = -1
            it = i
            while(found_before < 0 and it > 0):
                if(intersections[it-1].coords[0] in points):
                    found_before = it-1
                    idx_before = points.index(intersections[found_before].coords[0])
                    point_before_f = (val_x[idx_before],val_y[idx_before])
                    V[idx_v].t_before_f = keys[points.index(point_before_f)]
                else:
                    it = it-1
                    
            it = i + 1
            while(found_after < 0 and it < len(intersections)):
                if(intersections[it].coords[1] in points):
                    found_after = it
                    idx_after = points.index(intersections[found_after].coords[1])
                    point_after_f = (intersections[found_after].coords[1][0],intersections[found_after].coords[1][1])
                    V[idx_v].t_after_f = keys[points.index(point_after_f)]
                    dx = (val_x[idx_after] - V[idx_v].point.pos_x)#/abs(val_x[idx_after] - V[idx_v].point.pos_x)
                    dy = (val_y[idx_after] - V[idx_v].point.pos_y)#/abs(val_y[idx_after] - V[idx_v].point.pos_y)
                    V[idx_v].df = IntervalVector([[dx,dx],[dy,dy]])
                    V[idx_v].compute_u()
                else:
                    it = it+1
         
# #####################  Create graph from vertices and update edges ##################### 
g = Graph(V,idx_list,gamma_plus)
g.UpdateEdges()
g.print_graph()

# ##################### Create separators from winding sets #####################
pixel = 0.05 # separators precision in image contractor
seps,back_seps,contour_sep = g.CreateAllSeps(world,gamma,gamma_plus,dt,pixel)

# ##################### SIVIA #####################
epsilon = 0.1 # sivia's precision
stack = deque([IntervalVector(world)])
res_y, res_in, res_out = [], [], []
lf = LargestFirst(epsilon/2.0)
k = 0

vibes.beginDrawing()
for wn in seps.keys():
    fig = VIBesFig('Winding set W' + str(wn))
    fig.set_properties(100, 100, 800, 800)
    fig.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
    stack = deque([IntervalVector(world)])
    res_y, res_in, res_out = [], [], []
    lf = LargestFirst(epsilon/2.0)
    k = 0
    while len(stack) > 0:
        X = stack.popleft()
        k = k+1
        nb_in = Interval(0)
        sep = seps[wn]
        x_in, x_out = X.copy(),X.copy()
        sep.separate(x_in, x_out)
        if(x_in[0].is_empty() or x_in[1].is_empty()):  #it means that box is inside completely
            fig.draw_box(X,"gray[green]")
        elif(not (x_out[0].is_empty() or x_out[1].is_empty())): #partially inside
            x_in, x_out = X.copy(),X.copy()
            # contour_sep.separate(x_in, x_out)
            if(X.max_diam() < epsilon or x_in[0].is_empty() or x_in[1].is_empty()):
                fig.draw_box(X,"gray[yellow]")
            else:
                (X1, X2) = lf.bisect(X)
                stack.append(X1)
                stack.append(X2)
        else:
            fig.draw_box(X,"gray[blue]")

count = 0
for back_sep in back_seps:
    fig = VIBesFig('Gamma Minus ' + str(count))
    fig.set_properties(100, 100, 800, 800)
    fig.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
    stack = deque([IntervalVector(world)])
    res_y, res_in, res_out = [], [], []
    lf = LargestFirst(epsilon/2.0)
    k = 0
    while len(stack) > 0:
        X = stack.popleft()
        k = k+1
        nb_in = Interval(0)

        sep = back_sep
        x_in, x_out = X.copy(),X.copy()
        sep.separate(x_in, x_out)
        if(x_in[0].is_empty() or x_in[1].is_empty()):  #it means that box is inside completely
            fig.draw_box(X,"gray[green]")
        elif(not (x_out[0].is_empty() or x_out[1].is_empty())): #partially inside
            x_in, x_out = X.copy(),X.copy()
            # contour_sep.separate(x_in, x_out)
            if(X.max_diam() < epsilon or x_in[0].is_empty() or x_in[1].is_empty()):
                fig.draw_box(X,"gray[yellow]")
            else:
                (X1, X2) = lf.bisect(X)
                stack.append(X1)
                stack.append(X2)
        else:
            fig.draw_box(X,"gray[blue]")
    count += 1


fig_siv = VIBesFigMap("SIVIA")
fig_siv.set_properties(100, 100, 800, 800)
fig_siv.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
stack = deque([IntervalVector(world)])
res_y, res_in, res_out = [], [], []
lf = LargestFirst(epsilon/2.0)
k = 0
while len(stack) > 0:
    X = stack.popleft()
    k = k+1
    nb_in = Interval(0)

    for wn in seps.keys():
        sep = seps[wn]
        x_in, x_out = X.copy(),X.copy()
        sep.separate(x_in, x_out)
        if(x_in[0].is_empty() or x_in[1].is_empty()):  #it means that box is inside completely
            nb_in += Interval(1)
        elif(not (x_out[0].is_empty() or x_out[1].is_empty())): #partially inside
            nb_in += Interval(0,1)

    if(len(back_seps) > 0):
        for back_sep in back_seps:
            sep = back_sep
            x_in, x_out = X.copy(),X.copy()
            sep.separate(x_in, x_out)
            if(x_in[0].is_empty() or x_in[1].is_empty()):  #it means that box is inside completely
                nb_in += Interval(1)
            elif(not (x_out[0].is_empty() or x_out[1].is_empty())): #partially inside
                nb_in += Interval(0,1)

    x_in, x_out = X.copy(),X.copy()
    # contour_sep.separate(x_in, x_out)
    if(nb_in.ub() > nb_in.lb()):
        if(X.max_diam() < epsilon or x_in[0].is_empty() or x_in[1].is_empty()):
            fig_siv.draw_box(X,"k[k]")
        else:
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
vibes.endDrawing()