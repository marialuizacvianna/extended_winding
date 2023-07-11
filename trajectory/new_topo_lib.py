# from pyibex import *
# from pyibex.image import CtcRaster
from codac import *
from codac.unsupported import *
import numpy as np
from math import ceil,floor
import cv2
import matplotlib.pyplot as plt

class Point:
    def __init__(self,_pos_x,_pos_y):
        self.pos_x = _pos_x
        self.pos_y = _pos_y
    def coords(self):
        return (self.pos_x,self.pos_y)


class Vertice:
    def __init__(self,_pos_x,_pos_y,_nb):
        self.point = Point(_pos_x,_pos_y)
        self.t_before_i = 0
        self.t_after_i = 0
        self.t_before_f = 0
        self.t_after_f = 0
        self.di = 0
        self.df = 0
        self.u = 0
        self.nb = _nb
        self.e_i= []
        self.e_f = []

    def compute_u(self):
        dir = self.di[0]*self.df[1] - self.df[0]*self.di[1]

        if(dir.lb() > 0):
            self.u = -1
        elif(dir.ub() < 0):
            self.u = 1

    def print_vertice(self):
        print(f"pos_x = {self.point.pos_x}, pos_y = {self.point.pos_y}, di = {self.di}, df = {self.df}, u = {self.u}")
        print(f"self.t_before_i = {self.t_before_i}, self.t_after_i = {self.t_after_i},self.t_before_f = {self.t_before_f}, self.t_after_f = {self.t_after_f}")
  

class Edge:
    def __init__(self,_traj,_v_i,_v_f,_u,_nb):
        self.traj = _traj
        self.l_v = 0 #left value
        self.r_v = 0 #right value
        self.v_i = _v_i #initial vertice
        self.v_f = _v_f  #final vertice
        self.u = _u #update indice
        self.nb = _nb



class Graph:
    def __init__(self,_V,idx_list,gamma,_back_timer,_back_timel):
        self.V = _V
        self.min_wn = float('inf')
        self.max_wn = 0
        self.E = []
        idx_used = []
        count_edge = 0
        self.back_timer = _back_timer
        self.back_timel = _back_timel

        for i in range(len(idx_list) - 1):
            idx = idx_list[i]
            next_idx = idx_list[i+1]
            u = -self.V[idx].u
            t_begin = 0
            t_end = 0
            
            if(idx not in idx_used):
                idx_used.append(idx)
                u = self.V[idx].u
                t_begin = self.V[idx].t_before_i
            else:
                t_begin = self.V[idx].t_before_f

            if(next_idx not in idx_used):
                t_end = self.V[next_idx].t_after_i
            else:
                t_end = self.V[next_idx].t_after_f  
            
            new_t = Interval(t_begin,t_end)
            new_traj = TrajectoryVector(gamma)
            traj = new_traj.truncate_tdomain(new_t)
            new_e = Edge([traj],self.V[idx],self.V[next_idx],u,count_edge)
            self.V[idx].e_i.append(count_edge)
            self.V[next_idx].e_f.append(count_edge)
            self.E.append(new_e)
            count_edge+= 1

        if(len(self.V) == 0):
            self._min_wn = 1
            self._max_wn = 1
            traj = TrajectoryVector(gamma)
            pos_x = traj(gamma.tdomain().lb())[0]
            pos_y = traj(gamma.tdomain().lb())[1]
            new_v = Vertice(pos_x,pos_y,0)
            new_e = Edge([traj],new_v,new_v,1,count_edge)
            self.E.append(new_e)
            self.V.append(new_v)
            self.V[0].e_i.append(count_edge)
            self.V[0].e_f.append(count_edge)
            count_edge+= 1
        else:
            #last traj
            idx = idx_list[-1]
            next_idx = idx_list[0]
            u = -self.V[idx].u
            t_begin = self.V[idx].t_before_f
            t_end = self.V[next_idx].t_after_i

            new_t_1 = Interval(t_begin,gamma.tdomain().ub())
            new_t_2 = Interval(gamma.tdomain().lb(),t_end)
            new_traj_1 = TrajectoryVector(gamma)
            new_traj_2 = TrajectoryVector(gamma)
            traj_1 = new_traj_1.truncate_tdomain(new_t_1)
            traj_2 = new_traj_2.truncate_tdomain(new_t_2)
            new_e = Edge([traj_1,traj_2],self.V[idx],self.V[next_idx],u,count_edge)
            self.V[idx].e_i.append(count_edge)
            self.V[next_idx].e_f.append(count_edge)
            self.E.append(new_e)
            count_edge+= 1



    def print_graph(self):
        print("edges = ")
        for e in self.E:
            print("l_v = ", e.l_v)
            print("r_v = ", e.r_v)
            print("u = ", e.u)

    def UpdateEdges(self):
        min_val = 0
        for i in range(len(self.E)):
            e = self.E[i]
           
            if(i == 0):
                if(e.u == 1):
                    e.l_v = 1
                else:
                    e.r_v = -1
                    min_val = -1

            else:
                e_b = self.E[i-1]
                e.l_v = e_b.l_v + e.u
                e.r_v = e_b.r_v + e.u
                if(e.l_v < min_val):
                    min_val = e.l_v
                if(e.r_v < min_val):
                    min_val = e.r_v

        for e in self.E:
            if(min_val < 0):
                e.l_v += -min_val
                e.r_v += -min_val
            if(e.l_v < self.min_wn and e.l_v > 0):
                self.min_wn = e.l_v
            if(e.r_v < self.min_wn and e.r_v > 0):
                self.min_wn = e.r_v
            if(e.l_v > self.max_wn):
                self.max_wn = e.l_v
            if(e.r_v > self.max_wn):
                self.max_wn = e.r_v
                
    def CreateSep(self,wn,X,dt,eps,img_count):
        print("wn = ",wn)
        pixel_x = eps
        pixel_y = -eps
        npx = int((X[0].ub() - X[0].lb())/abs(pixel_x))
        npy = int((X[1].ub() - X[1].lb())/abs(pixel_y))
        img_aux = np.zeros((npx, npy), dtype=np.int64)

        e = self.E[0]
        nxt_idx = 1
        edge_out = []
        total_len = 0
        
        while(e.l_v != wn and nxt_idx < len(self.E)):
            edge_out.append(e.nb)
            total_len += 1
            e = self.E[nxt_idx]
            nxt_idx += 1

        if(e.l_v != wn):
            return
        
        edge_in = [[]]

        img_aux = np.zeros((npx, npy), dtype=np.int64)
        imgs = []
        while(total_len < len(self.E)):
            edge_in[len(edge_in)-1].append(e.nb)
            total_len += 1
            for traj in e.traj:
                domain = traj.tdomain()
                t = domain.lb()
                last_t = domain.lb()
                while(t < domain.ub()):
                    t += dt
                    if(t > domain.ub()):
                        t = domain.ub()

                    xi = traj(Interval(last_t,t))[0]
                    yi = traj(Interval(last_t,t))[1]

                    x_pix = (xi - X[0].lb())/pixel_x
                    y_pix= (yi - X[1].ub())/pixel_y

                    for i in range(floor(x_pix.lb()),ceil(x_pix.ub())):
                        for j in range(floor(y_pix.lb()),ceil(y_pix.ub())):
                            img_aux[i,j] = 1

                    last_t = t

            if(self.E[edge_in[len(edge_in)-1][0]].v_i.nb == self.E[edge_in[len(edge_in)-1][len(edge_in[len(edge_in)-1]) - 1]].v_f.nb):
                edge_in.append([])
               
                imgs.append(img_aux)
                img_aux = np.zeros((npx, npy), dtype=np.int64)

                found = 0
                e = self.E[0]
                nxt_idx = 1
                while(not found and nxt_idx < len(self.E)):
                    if(e.l_v == wn):
                        found = 1
                        for l in edge_in:
                            if(e.nb in l):
                                found = 0
                    else:
                        if(not e.nb in edge_out):
                            edge_out.append(e.nb)
                            total_len += 1
                        found = 0

                    if(not found):
                        e = self.E[nxt_idx]
                        nxt_idx += 1

            else:
                l_v_f = self.V[e.v_f.nb].e_i
                for idx_et in l_v_f:
                    et = self.E[idx_et]
                    if(et.l_v == wn):
                        e = et
                    else:
                        if(not et.nb in edge_out):
                            edge_out.append(et.nb)
                            total_len += 1

        images_out = []
        for img_aux in imgs:
            img_count += img_aux
            img_count[img_count > 0] = 1
            contours, hyera = cv2.findContours(np.ascontiguousarray(img_aux.copy(), dtype=np.uint8).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            max_area = 0
            c = -1
            idx = 0
            for cntr in contours:
                if(hyera[0][idx][3] == -1):
                    area = cv2.contourArea(cntr)
                    if(area > max_area):
                        c = cntr
                        max_area = area
                idx +=1
            img_out_i = np.ascontiguousarray(img_aux.copy(), dtype=np.uint8)
            if(max_area > 0):
                cv2.drawContours(img_out_i,[c],contourIdx=0, color=(255,255,255),thickness=-1)
            images_out.append(img_out_i)

        img_out = np.ascontiguousarray(np.zeros((npx, npy), dtype=np.int64), dtype=np.uint8)
        for img in images_out:
            img[img > 0] = 1
            img_out += img
            img_out[(img_out%2) == 0] = 0

        img_in = np.ones((npx, npy), dtype=np.uint8) - img_out
        for img_aux in imgs:
            img_out[img_aux > 0 ] = 1
            img_in[img_aux > 0 ] = 1
        
        img_out = img_out.cumsum(0).cumsum(1)
        img_in = img_in.cumsum(0).cumsum(1)
        ctcOut = CtcRaster(img_out, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
        ctcIn = CtcRaster(img_in, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
        sep = SepCtcPair(ctcIn, ctcOut)
        return img_count,sep

    def CreateBackSeps(self,X,gamma,gamma_pos,dt,eps):
        pixel_x = eps
        pixel_y = -eps
        npx = int((X[0].ub() - X[0].lb())/abs(pixel_x))
        npy = int((X[1].ub() - X[1].lb())/abs(pixel_y))

        seps = []
        left_interval = (self._tdomain/4.0) + 2*((self._tdomain/4.0).diam())
        for domain in self.back_timel:
            img_aux = np.zeros((npx, npy), dtype=np.int64)
            t = domain.lb()
            last_t = domain.lb()
            while(t < domain.ub()):
                t += dt
                if(t > domain.ub()):
                    t = domain.ub()


                xi = gamma(Interval(left_interval.ub()) - Interval(last_t,t))[0]
                yi = gamma(Interval(left_interval.ub()) - Interval(last_t,t))[1]

                x_pix = (xi - X[0].lb())/pixel_x
                y_pix= (yi - X[1].ub())/pixel_y

                for i in range(floor(x_pix.lb()),ceil(x_pix.ub()) + 1):
                    for j in range(floor(y_pix.lb()),ceil(y_pix.ub()) +1 ):
                        img_aux[i,j] = 1
                

                xi = gamma_pos(Interval(left_interval.ub()) - Interval(last_t,t))[0]
                yi = gamma_pos(Interval(left_interval.ub()) - Interval(last_t,t))[1]

                x_pix = (xi - X[0].lb())/pixel_x
                y_pix= (yi - X[1].ub())/pixel_y

                for i in range(floor(x_pix.lb()),ceil(x_pix.ub()) +1 ):
                    for j in range(floor(y_pix.lb()),ceil(y_pix.ub()) +1):
                        img_aux[i,j] = 1

                last_t = t
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

            img_out = img_out.cumsum(0).cumsum(1)
            img_in = img_in.cumsum(0).cumsum(1)
            ctcOut = CtcRaster(img_out, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
            ctcIn = CtcRaster(img_in, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
            sep = SepCtcPair(ctcIn, ctcOut)
            seps.append(sep)

        for domain in self.back_timer:
            img_aux = np.zeros((npx, npy), dtype=np.int64)
            t = domain.lb()
            last_t = domain.lb()
            while(t < domain.ub()):
                t += dt
                if(t > domain.ub()):
                    t = domain.ub()

                xi = gamma(Interval(last_t,t))[0]
                yi = gamma(Interval(last_t,t))[1]

                x_pix = (xi - X[0].lb())/pixel_x
                y_pix= (yi - X[1].ub())/pixel_y

                for i in range(floor(x_pix.lb()),ceil(x_pix.ub()) + 1):
                    for j in range(floor(y_pix.lb()),ceil(y_pix.ub()) + 1):
                        img_aux[i,j] = 1

                xi = gamma_pos(Interval(last_t,t))[0]
                yi = gamma_pos(Interval(last_t,t))[1]

                x_pix = (xi - X[0].lb())/pixel_x
                y_pix= (yi - X[1].ub())/pixel_y

                for i in range(floor(x_pix.lb()),ceil(x_pix.ub()) + 1):
                    for j in range(floor(y_pix.lb()),ceil(y_pix.ub()) +1):
                        img_aux[i,j] = 1

                last_t = t

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

            img_out = img_out.cumsum(0).cumsum(1)
            img_in = img_in.cumsum(0).cumsum(1)
            ctcOut = CtcRaster(img_out, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
            ctcIn = CtcRaster(img_in, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
            sep = SepCtcPair(ctcIn, ctcOut)
            seps.append(sep)
        return seps


    def CreateContourSep(self,X,eps,img_aux):
        pixel_x = eps
        pixel_y = -eps
        npx = int((X[0].ub() - X[0].lb())/abs(pixel_x))
        npy = int((X[1].ub() - X[1].lb())/abs(pixel_y))

        img_out = img_aux.copy()
        img_in = np.ones((npx, npy), dtype=np.uint8) - img_out

        img_out = img_out.cumsum(0).cumsum(1)
        img_in = img_in.cumsum(0).cumsum(1)

        ctcOut = CtcRaster(img_out, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
        ctcIn = CtcRaster(img_in, X[0].lb(), X[1].ub(), pixel_x, pixel_y)
        sep = SepCtcPair(ctcIn, ctcOut)
        return sep

    def CreateAllSeps(self,X,gamma,gamma_pos,dt,eps):
            if(self.max_wn <= 0):
                return []
            seps = dict()
            back_sep = []
            pixel_x = eps
            pixel_y = -eps
            npx = int((X[0].ub() - X[0].lb())/abs(pixel_x))
            npy = int((X[1].ub() - X[1].lb())/abs(pixel_y))

            img_aux = np.zeros((npx, npy), dtype=np.int64)

            for i in np.arange(self.min_wn,self.max_wn+1,1):
                img_aux,seps[i] = self.CreateSep(i,X,dt,eps,img_aux)

            if(len(self.back_timer) > 0 or len(self.back_timel)):
                back_sep = self.CreateBackSeps(X,gamma,gamma_pos,dt,eps)

            contour_sep = self.CreateContourSep(X,eps,img_aux)

            return seps,back_sep,contour_sep
