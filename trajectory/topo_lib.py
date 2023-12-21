# from pyibex import *
# from pyibex.image import CtcRaster
from codac import *
from codac.unsupported import *
import numpy as np
from math import ceil,floor
import cv2
import matplotlib.pyplot as plt



class Graph:
    def __init__(self,inter_list,tdomain,d_list,back_timer,back_timel):
        self._n_v = len(inter_list)
        self._n_e = 2*self._n_v
        self._V = []
        self._E = []
        self._tdomain = tdomain
        self._min_wn = float('inf')
        self._max_wn = 0
        self._update = 0
        self._back_timer = back_timer
        self._back_timel = back_timel
        self._v_count = 0
        self._e_count = 0

        if(self._n_v > 0):
            for i in range(self._n_v):
                self.AddVertice(inter_list[i][0],inter_list[i][1],d_list[0][i],d_list[1][i])
        else:
            self._min_wn = 1
            self._max_wn = 1
            self._n_v = 1
            self._n_e = 1
            self.AddVertice(Interval(tdomain.lb()),Interval(tdomain.ub()),[Interval(0),Interval(1)],[Interval(1),Interval(0)])

        self.EdgesFromVertices()

    def print_graph(self):
        for i in range(self._n_v):
            print("_V[i]._tdomain = ", self._V[i]._tdomain)

        print("edges = ")
        for i in range(self._n_e):
            print("i = ",i)
            print("g._E[i]._T = ",self._E[i]._T)
            print("g._E[i]._l_v = ", self._E[i]._l_v)
            print("g._E[i]._r_v = ", self._E[i]._r_v)
            print("g._E[i]._u = ", self._E[i]._u)

    def EdgesFromVertices(self):
        V_dict = dict()
        for v in self._V:
            V_dict[v._t_i.lb()] = v
            V_dict[v._t_f.lb()] = v

        V_keys = list(V_dict.keys())
        V_keys.sort()

        if(self._n_e > 1):
            for i in range(self._n_e):
                vi = V_dict[V_keys[i]]
                vf = 0
                t = Interval(0)
                if(i == self._n_e - 1):
                    t = [Interval(V_dict[V_keys[len(V_keys) - 1]]._t_f.lb(),self._tdomain.ub()),Interval(self._tdomain.lb(),V_dict[V_keys[0]]._t_i.ub())]
                    vf = V_dict[V_keys[0]]
                else:
                    t_i = 0
                    t_f = 0
                
                    t_i = V_keys[i]
                    if(V_dict[V_keys[i+1]]._t_i.lb() == V_keys[i+1]):
                        t_f = V_dict[V_keys[i+1]]._t_i.ub()
                    else:
                        t_f = V_dict[V_keys[i+1]]._t_f.ub()
                
                    t = [Interval(t_i,t_f)]
                    vf = V_dict[V_keys[i+1]]

                dir = vi._d_i[0]*vi._d_f[1] - vi._d_f[0]*vi._d_i[1]
            
                if(not((vi._t_f&Interval(t[0].lb())).is_empty())):
                    dir = -dir

                new_e = self.AddEdge(t,vi,vf,dir)


                self._V[vi._nb]._e_i.append(new_e)
                self._V[vf._nb]._e_f.append(new_e)

        else:
            new_e = self.AddEdge([self._tdomain],self._V[0],self._V[0],Interval(-1))
            self._V[0]._e_i.append(new_e)
            self._V[0]._e_f.append(new_e)


    def AddVertice(self,t_1,t_2,d_i,d_f):
        new_v = Vertice(t_1,t_2,d_i,d_f,self._v_count)
        self._V.append(new_v)
        self._v_count += 1
        return new_v

    def AddEdge(self,t,v_1,v_2,dir):
        new_e = Edge(t,v_1,v_2,self._e_count)
        if(dir.lb() > 0):
            new_e._u = -1
        elif(dir.ub() < 0):
            new_e._u = 1
       
        self._E.append(new_e)
        self._e_count += 1
        return new_e

    def UpdateEdges(self):
        min_val = 0
        for i in range(self._n_e):
            e = self._E[i]
           
            if(i == 0):
                if(e._u == 1):
                    e._l_v = 1
                else:
                    e._r_v = -1
                    min_val = -1

            else:
                e_b = self._E[i-1]
                e._l_v = e_b._l_v + e._u
                e._r_v = e_b._r_v + e._u
                if(e._l_v < min_val):
                    min_val = e._l_v
                if(e._r_v < min_val):
                    min_val = e._r_v

        for e in self._E:
            if(min_val < 0):
                e._l_v += -min_val
                e._r_v += -min_val
            if(e._l_v < self._min_wn and e._l_v > 0):
                self._min_wn = e._l_v
            if(e._r_v < self._min_wn and e._r_v > 0):
                self._min_wn = e._r_v
            if(e._l_v > self._max_wn):
                self._max_wn = e._l_v
            if(e._r_v > self._max_wn):
                self._max_wn = e._r_v

        self._update = 1

    def CreateSep(self,wn,X,gamma,dt,eps,img_count):
        pixel_x = eps
        pixel_y = -eps
        npx = int((X[0].ub() - X[0].lb())/abs(pixel_x))
        npy = int((X[1].ub() - X[1].lb())/abs(pixel_y))
        img_aux = np.zeros((npx, npy), dtype=np.int64)

        e = self._E[0]
        nxt_idx = 1
        edge_out = []
        total_len = 0

        while(e._l_v != wn and nxt_idx < len(self._E)):
            edge_out.append(e._nb)
            total_len += 1
            e = self._E[nxt_idx]
            nxt_idx += 1

        if(e._l_v != wn):
            return

        edge_in = [[]]

        img_aux = np.zeros((npx, npy), dtype=np.int64)
        imgs = []
        while(total_len < len(self._E)):
            edge_in[len(edge_in)-1].append(e._nb)
            total_len += 1
            for domain in e._T:
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

                    for i in range(floor(x_pix.lb()),ceil(x_pix.ub())):
                        for j in range(floor(y_pix.lb()),ceil(y_pix.ub())):
                            img_aux[i,j] = 1

                    last_t = t

            if(self._E[edge_in[len(edge_in)-1][0]]._v_i == self._E[edge_in[len(edge_in)-1][len(edge_in[len(edge_in)-1]) - 1]]._v_f) :
                end_cycle = 1
                edge_in.append([])
               
                imgs.append(img_aux)
                img_aux = np.zeros((npx, npy), dtype=np.int64)

                found = 0
                e = self._E[0]
                nxt_idx = 1
                while(not found and nxt_idx < len(self._E)):
                    if(e._l_v == wn):
                        found = 1
                        for l in edge_in:
                            if(e._nb in l):
                                found = 0
                    else:
                        if(not e._nb in edge_out):
                            edge_out.append(e._nb)
                            total_len += 1
                        found = 0

                    if(not found):
                        e = self._E[nxt_idx]
                        nxt_idx += 1

            else:
               
                l_v_f = self._V[e._v_f._nb]._e_i
                for et in l_v_f:
                    if(et._l_v == wn):
                        e = et
                    else:
                        if(not et._nb in edge_out):
                            edge_out.append(et._nb)
                            total_len += 1

        images_out = []
        for img_aux in imgs:
            img_count += img_aux
            img_count[img_count > 0] = 1
            contours, hyera = cv2.findContours(np.ascontiguousarray(img_aux.copy(), dtype=np.uint8).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            index=0
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
        for domain in self._back_timel:
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

        for domain in self._back_timer:
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


    def CreateContourSep(self,X,dt,eps,img_aux):
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
            if(self._max_wn <= 0):
                return []
            seps = dict()
            back_sep = []
            pixel_x = eps
            pixel_y = -eps
            npx = int((X[0].ub() - X[0].lb())/abs(pixel_x))
            npy = int((X[1].ub() - X[1].lb())/abs(pixel_y))

            img_aux = np.zeros((npx, npy), dtype=np.int64)

            for i in np.arange(self._min_wn,self._max_wn+1,1):
                img_aux,seps[i] = self.CreateSep(i,X,gamma_pos,dt,eps,img_aux)

            if(len(self._back_timer) > 0 or len(self._back_timel)):
                back_sep = self.CreateBackSeps(X,gamma,gamma_pos,dt,eps)

            contour_sep = self.CreateContourSep(X,dt,eps,img_aux)
            return seps,back_sep,contour_sep

class Vertice:
    def __init__(self,t_1,t_2,d_i,d_f,nb):
        self._t_i = t_1  #interval t1
        self._t_f = t_2 #interval t2
        self._tdomain = Interval(t_1.lb(),t_2.ub())
        self._d_i = d_i
        self._d_f = d_f
        self._e_i= []
        self._e_f = []
        self._nb = nb

class Edge:
    def __init__(self,t,v_i,v_f,nb):
        self._T = t #list with values of times in gamma
        self._r_v = 0 #right value
        self._l_v = 0 #left value
        self._v_i = v_i #initial vertice
        self._v_f = v_f #final vertice
        self._u = 0 #update indice
        self._nb = nb



if __name__ == "__main__":
    print("vertices = ")
    for i in range(g._n_v):
        print("g._V[i]._tdomain = ", g._V[i]._tdomain)

        print("edges = ")
        for i in range(g._n_e):
            print("i = ",i)
            print("g._E[i]._T = ",g._E[i]._T)
            print("g._E[i]._l_v = ", g._E[i]._l_v)
            print("g._E[i]._r_v = ", g._E[i]._r_v)
            print("g._E[i]._u = ", g._E[i]._u)
