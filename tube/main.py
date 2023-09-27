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
x1_robot = "(-9.6* (t-1.8)^2  + 11.4*(0.5* (t-1.8)^3 - (t-1.8)) +30)"
dx1_robot = "(-19.2* (t-1.8) + 17.1* (t-1.8)^2 - 11.4)"
ddx1_robot = "(-19.2 + 2*17.1* (t-1.8))"
x2_robot = "(-9.6*(0.5* (t-1.8)^3 - (t-1.8)) - 11.4* (t-1.8)^2 + 30)"
dx2_robot = "(-14.4* (t-1.8)^2 +9.6 -22.8* (t-1.8))"
ddx2_robot = "(-14.4*2* (t-1.8) -22.8)"
#mission time interval
tdomain = Interval(0,3.6) 
#time step
dt=0.01
#Range of visibility on each side
L = 5.0
#Area to classify
world = IntervalVector([[-20,40],[-25,40]]) 
#size of the robot for visualization
robot_size = 3.

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

######
#separate gamma into gamma + and gamma 
d_theta = ((a_robot[1]*v_robot[0]) - (a_robot[0]*v_robot[1]))/(v_robot[0]*v_robot[0] + v_robot[1]*v_robot[1])
sin_theta = v_robot[1]/sqrt(v_robot[0]*v_robot[0] + v_robot[1]*v_robot[1])
cos_theta = v_robot[0]/sqrt(v_robot[0]*v_robot[0] + v_robot[1]*v_robot[1])
l_fpr = -(v_robot[0]*cos_theta + v_robot[1]*sin_theta)/(d_theta) 
l_fpl = Tube(l_fpr)
time_pfr = Interval(-oo,oo)
time_pfl = Interval(-oo,oo)
interval_right = []
interval_left = []
for i in range(l_fpr.nb_slices()):
    value = l_fpr.slice(i).codomain()
    t_value = l_fpr.slice(i).tdomain()
    l_fpr.slice(i).set(value & Interval(0,L))
    l_fpl.slice(i).set(value & Interval(-L,0))
    if(value != Interval(-oo,oo)):
        if(not (value & [0,L]).is_empty()):
            if(time_pfr != Interval(-oo,oo)):
                if(not (t_value & time_pfr).is_empty()):
                    time_pfr |= t_value
                else:
                    interval_right.append(time_pfr.copy())
                    time_pfr = t_value
            else:
                time_pfr = t_value
            print("direita")
            print("value = ",value)
            print("t_value = ",t_value)
        else:
            l_fpr.slice(i).set(Interval(L,L))
        if(not (value & [-L,0]).is_empty()):
            if(time_pfl != Interval(-oo,oo)):
                if(not (t_value & time_pfl).is_empty()):
                    time_pfl |= t_value
                else:
                    interval_left.append(time_pfl.copy())
                    time_pfl = t_value
            else:
                time_pfl = t_value
            print("left")
            print("value = ",value)
            print("t_value = ",t_value)
        else:
            l_fpl.slice(i).set(Interval(-L,-L))
    else:
        l_fpr.slice(i).set(Interval(L,L))
        l_fpl.slice(i).set(Interval(-L,-L))

    # print('l_fpr.slice(i) = ', l_fpr.slice(i))
    # print('l_fpr.slice(i).codomain() = ', l_fpr.slice(i).codomain())
    # print('l_fpr.slice(i).next_slice().codomain() = ', l_fpr.slice(i).next_slice().codomain())
if(time_pfl != Interval(-oo,oo)):
    interval_left.append(time_pfl.copy())
if(time_pfr != Interval(-oo,oo)):
    interval_right.append(time_pfr.copy())

gamma_plus = TubeVector(gamma)
v_plus = v.copy()
list_gamma_minus = []

print('interval_right = ',interval_right)
print('interval_left = ',interval_left)
if(len(interval_right) > 0 or len(interval_left) > 0):
    x_right_plus= TubeVector(tdomain,dt,2)
    v_right_plus = TubeVector(tdomain,dt,2)
    x_right_plus[0] = x[0] + l_fpr*sin_theta
    x_right_plus[1] = x[1] - l_fpr*cos_theta
    v_right_plus[0] = v_robot[0] + l_fpr*cos_theta
    v_right_plus[1] = v_robot[1] + l_fpr*sin_theta
    v_plus[0] = v_right_plus
    x_left_plus= TubeVector(tdomain,dt,2)
    v_left_plus = TubeVector(tdomain,dt,2)
    x_left_plus[0] = x[0] + l_fpl*sin_theta
    x_left_plus[1] = x[1] - l_fpl*cos_theta
    v_left_plus[0] = v_robot[0] + l_fpl*cos_theta
    v_left_plus[1] = v_robot[1] + l_fpl*sin_theta
    v_plus[1] = v_left_plus
    gamma_plus = ConcatenateTubes([x_right_plus,x_rl,InverseTube(x_left_plus,tdomain,dt),x_lr],dt)
    for ti in interval_right:
        # res = TubeVector(Interval(ti.lb(),ti.lb()+2*ti.diam()),dt,2)
        # print("x_right = ",x_right)
        # print("ti = ",ti)
        p1 = TubeVector(x_right)
        p1.truncate_tdomain(Interval(round(ti.lb(),2),round(ti.ub(),2)))
        p2 = TubeVector(x_right_plus).truncate_tdomain(Interval(round(ti.lb(),2),round(ti.ub(),2)))
        list_gamma_minus.append(ConcatenateTubes([InverseTube(p1,p1[0].tdomain(),dt),p2],dt))
    for ti in interval_left:
        # res = TubeVector(Interval(ti.lb(),ti.lb()+2*ti.diam()),dt,2)
        # print("x_right = ",x_right)
        # print("ti = ",ti)
        p1 = TubeVector(x_left)
        p1.truncate_tdomain(Interval(round(ti.lb(),2),round(ti.ub(),2)))
        p2 = TubeVector(x_left_plus).truncate_tdomain(Interval(round(ti.lb(),2),round(ti.ub(),2)))
        list_gamma_minus.append(ConcatenateTubes([InverseTube(p1,p1[0].tdomain(),dt),p2],dt))

print("len(list_gamma_minus) = ",len(list_gamma_minus))   
######
#find self-intersections in gamma_plus
tplane = TPlane(gamma_plus.tdomain())
tplane.compute_detections(5*dt, gamma_plus)
tplane.compute_proofs(gamma_plus)
loops = tplane.proven_loops()

######
#derivatives in self-intersections 
d_list_i,d_list_f = TangentLoop(v_plus,tdomain,loops)

######
#create graph and update edges
g = Graph(loops,gamma_plus.tdomain(),[d_list_i,d_list_f],[],[])
g.UpdateEdges()

######
#Graphics with Vibes 
# beginDrawing()
# fig_map = VIBesFigMap("Map")
# # fig_map.smooth_tube_drawing(True)
# fig_map.set_tube_max_disp_slices(10000)
# fig_map.set_properties(100, 100, 800, 800)
# fig_map.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
# fig_map.add_trajectory(x_truth, "x", 0, 1,"red")
# fig_map.add_tube(gamma, "[gamma]", 0, 1)
# for l in loops:
#     fig_map.draw_box(gamma_plus(l[0]),"k[]")
#     fig_map.draw_box(gamma_plus(l[1]),"k[]")
# fig_map.show(0.)

beginDrawing()
fig_map = VIBesFigMap("Map")
fig_map.smooth_tube_drawing(True)
fig_map.set_tube_max_disp_slices(10000)
fig_map.set_properties(100, 100, 800, 800)
# fig_map.axis_limits(-13,13,-16,9)
fig_map.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
# fig_map.add_trajectory(x_truth, "x", 0, 1,"red")
fig_map.add_tube(x, "[x]", 0, 1)


# fig_map.add_tube(x_right, "[x_right]", 0, 1)
# fig_map.add_tube(x_right_plus, "[x_right_plus]", 0, 1)
# fig_map.set_tube_color(x_right,"darkGray[darkGray]")

# fig_map.add_tube(x_left, "[x_left]", 0, 1)
# fig_map.add_tube(x_left_plus, "[x_left_plus]", 0, 1)
# fig_map.set_tube_color(x_left,"darkGray[darkGray]")

# fig_map.add_tube(x_rl, "[x_rl]", 0, 1)
# fig_map.set_tube_color(x_rl,"darkGray[darkGray]")
# fig_map.add_tube(x_lr, "[x_lr]", 0, 1)
# fig_map.set_tube_color(x_lr,"darkGray[darkGray]")

fig_map.add_tube(gamma_plus, "[gamma_plus]", 0, 1)
fig_map.set_tube_color(gamma_plus,"darkGray[darkGray]")

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
fig_map.draw_vehicle([x_truth[0](tdomain.ub()),x_truth[1](tdomain.ub()),atan2(dx_robot[1](tdomain.ub()),dx_robot[0](tdomain.ub()))], robot_size)
fig_map.show(robot_size)




fig_fpr = VIBesFigTube("FPR")
# fig_fpr.smooth_tube_drawing(True)
# fig_fpr.set_tube_max_disp_slices(10000)
# fig_fpr.set_properties(100, 100, 800, 800)
# fig_fpr.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
# fig_fpr.add_tube(Tube(tdomain, Interval(-L,L)), "limit1","red[]")
fig_fpr.add_tube(l_fpr, "[l_fpr]","k[k]")
fig_fpr.add_tube(l_fpl, "[l_fpl]","red[red]")
fig_fpr.show()


# ######
# #Create separators from winding sets
# pixel = 0.05 # separators precision in image contractor
# seps,back_seps,contour_sep = g.CreateAllSeps(world,gamma,gamma_plus,dt,pixel)

# ######
# #SIVIA 
# epsilon = 0.1 # sivia's precision
# stack = deque([IntervalVector(world)])
# res_y, res_in, res_out = [], [], []
# lf = LargestFirst(epsilon/2.0)
# k = 0

# vibes.beginDrawing()
# for wn in seps.keys():
#     fig = VIBesFig('Winding set W' + str(wn))
#     fig.set_properties(100, 100, 800, 800)
#     fig.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
#     stack = deque([IntervalVector(world)])
#     res_y, res_in, res_out = [], [], []
#     lf = LargestFirst(epsilon/2.0)
#     k = 0
#     while len(stack) > 0:
#         X = stack.popleft()
#         k = k+1
#         nb_in = Interval(0)
#         sep = seps[wn]
#         x_in, x_out = X.copy(),X.copy()
#         sep.separate(x_in, x_out)
#         if(x_in[0].is_empty() or x_in[1].is_empty()):  #it means that box is inside completely
#             fig.draw_box(X,"gray[green]")
#         elif(not (x_out[0].is_empty() or x_out[1].is_empty())): #partially inside
#             x_in, x_out = X.copy(),X.copy()
#             contour_sep.separate(x_in, x_out)
#             if(X.max_diam() < epsilon or x_in[0].is_empty() or x_in[1].is_empty()):
#                 fig.draw_box(X,"gray[yellow]")
#             else:
#                 (X1, X2) = lf.bisect(X)
#                 stack.append(X1)
#                 stack.append(X2)
#         else:
#             fig.draw_box(X,"gray[blue]")

# count = 0
# for back_sep in back_seps:
#     fig = VIBesFig('Gamma Minus ' + str(count))
#     fig.set_properties(100, 100, 800, 800)
#     fig.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
#     stack = deque([IntervalVector(world)])
#     res_y, res_in, res_out = [], [], []
#     lf = LargestFirst(epsilon/2.0)
#     k = 0
#     while len(stack) > 0:
#         X = stack.popleft()
#         k = k+1
#         nb_in = Interval(0)

#         sep = back_sep
#         x_in, x_out = X.copy(),X.copy()
#         sep.separate(x_in, x_out)
#         if(x_in[0].is_empty() or x_in[1].is_empty()):  #it means that box is inside completely
#             fig.draw_box(X,"gray[green]")
#         elif(not (x_out[0].is_empty() or x_out[1].is_empty())): #partially inside
#             x_in, x_out = X.copy(),X.copy()
#             contour_sep.separate(x_in, x_out)
#             if(X.max_diam() < epsilon or x_in[0].is_empty() or x_in[1].is_empty()):
#                 fig.draw_box(X,"gray[yellow]")
#             else:
#                 (X1, X2) = lf.bisect(X)
#                 stack.append(X1)
#                 stack.append(X2)
#         else:
#             fig.draw_box(X,"gray[blue]")
#     count += 1

vibes.endDrawing()

# fig_siv = VIBesFigMap("SIVIA")
# fig_siv.set_properties(100, 100, 800, 800)
# fig_siv.axis_limits(world[0].lb(),world[0].ub(),world[1].lb(),world[1].ub())
# stack = deque([IntervalVector(world)])
# res_y, res_in, res_out = [], [], []
# lf = LargestFirst(epsilon/2.0)
# k = 0
# while len(stack) > 0:
#     X = stack.popleft()
#     k = k+1
#     nb_in = Interval(0)

#     for wn in seps.keys():
#         sep = seps[wn]
#         x_in, x_out = X.copy(),X.copy()
#         sep.separate(x_in, x_out)
#         if(x_in[0].is_empty() or x_in[1].is_empty()):  #it means that box is inside completely
#             nb_in += Interval(1)
#         elif(not (x_out[0].is_empty() or x_out[1].is_empty())): #partially inside
#             nb_in += Interval(0,1)

#     if(len(back_seps) > 0):
#         for back_sep in back_seps:
#             sep = back_sep
#             x_in, x_out = X.copy(),X.copy()
#             sep.separate(x_in, x_out)
#             if(x_in[0].is_empty() or x_in[1].is_empty()):  #it means that box is inside completely
#                 nb_in += Interval(1)
#             elif(not (x_out[0].is_empty() or x_out[1].is_empty())): #partially inside
#                 nb_in += Interval(0,1)

#     x_in, x_out = X.copy(),X.copy()
#     contour_sep.separate(x_in, x_out)
#     if(nb_in.ub() > nb_in.lb()):
#         if(X.max_diam() < epsilon or x_in[0].is_empty() or x_in[1].is_empty()):
#             fig_siv.draw_box(X,"k[k]")
#         else:
#             (X1, X2) = lf.bisect(X)
#             stack.append(X1)
#             stack.append(X2)
#     else:
#         if((nb_in) == Interval(0)):
#             fig_siv.draw_box(X,"gray[w]")
#         elif((nb_in) == Interval(1)):
#             fig_siv.draw_box(X,"gray[#CCFF99]")
#         elif((nb_in) == Interval(2)):
#             fig_siv.draw_box(X,"gray[#CCE5FF]")
#         elif((nb_in) == Interval(3)):
#             fig_siv.draw_box(X,"gray[pink]")
#         elif((nb_in) == Interval(4)):
#             fig_siv.draw_box(X,"gray[gray]")
#         elif((nb_in) == Interval(5)):
#             fig_siv.draw_box(X,"gray[orange]")

# total_time_end = datetime.datetime.now()

# print('total time = '+str((total_time_end - total_time_begin).total_seconds())+"s")

