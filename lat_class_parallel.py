import numpy as np
import scipy as scp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

import os
import glob
import time
import IPython.display

from itertools import product


class lattice_sim:
    """
    Lattice simulation class object
    For MCMC simulations of the dualized kl lattice formulation
    the k flux constraint requires a specialized sampling method
    using the worm algorithm
    """


    def __init__(self):
        """Lattice variables"""

        ##chemical potential mu
        self.mu = 0.90

        ##size of the lattice
        nt = 10
        nx = 3
        ny = 3
        nz = 4
        #lat_size = [nt, nx, ny, nz]
        self.lat_size = [nx, ny]
        self.dim = len(self.lat_size)
        self.num_sites = np.prod(self.lat_size)


        ##size of one link field on the lattice
        self.link_size = np.concatenate(([len(self.lat_size)], self.lat_size))

        ##size of one lattice conf
        conf_dof = 2
        self.conf_size = np.concatenate(([conf_dof], [len(self.lat_size)], self.lat_size))
        print(self.conf_size)

        ##initialize with zeros
        self.lat_links = np.zeros(shape=self.conf_size, dtype=np.int64)
        self.l_links = np.zeros(shape=self.link_size, dtype=np.int64)
        self.k_links = np.zeros(shape=self.link_size, dtype=np.int64)

        ##an array saving the current status of the f function
        self.f = np.zeros(shape=self.lat_size, dtype=np.int64)

        """Read weights from file"""
        ##the weights whic hare needed for calculating probabilities
        #W = np.fromfile("weights.dat", sep=",")
        self.W = np.loadtxt("W.txt")

        #self.W = np.ones(10000)


        """Worm variables"""

        self.num_worms = 1

        ##head array keeps track of heads potential movements
        ##similar to tail array
        ##head[:,0]= old head
        ##head[:,1]= prop head
        ##head[:,2]= new head
        self.head = np.zeros((self.num_worms, 2, self.dim), dtype = int)
        for d in range(self.dim):
            self.head[:,0,d] = np.random.randint(0,self.lat_size[d], size = self.num_worms)
        """Does python create a reference or copy?"""
        """Update each time the head is modified!!!"""
        self.head[:,1] = self.head[:,0].copy()
        self.tail = self.head[:,0].copy()

        ##variables which descrbe a worms potential current move
        self.move_dim = np.zeros((self.num_worms), dtype = int)
        self.move_sign = np.zeros((self.num_worms), dtype = int)
        self.move_i = np.zeros((self.num_worms), dtype = int)
        self.moves = np.zeros((self.num_worms, self.dim), dtype = int)

        ##the change of k values which is picked randomly from [-1,1]
        self.k_delta = np.random.randint(low=0, high=2, size=self.num_worms)*2 - 1
        ###self.k_delta = np.random.randint(0,1)*2 - 1

        ##the lattice field alterations for each worm in the current move iteration
        self.dk = np.zeros((self.num_worms), dtype = int)
        self.df = np.zeros((self.num_worms), dtype = int)

        ##save the initial tweak of f for each worm at the start
        ##has to be added later
        self.df0 = np.zeros((self.num_worms), dtype = int)


        ##the current changes saved as a lattice "mask"
        self.k_change = np.zeros(self.link_size, dtype = int)
        self.f_change = np.zeros(self.lat_size, dtype = int)




        ##the link direction in which the worm is pointing at the moment
        self.worm_link_coord = np.zeros((self.num_worms, self.dim+1), dtype = int)
        self.worm_link_coord[:,1:] = self.head[:,0].copy()


        ##the viable moves the worm can make
        self.move_set = np.array([
                    [1,0],
                    [0,1],
                    [-1,0],
                    [0,-1],
        ])

        ##number of moves to pick from
        self.n_moves = len(self.move_set)

        ##array for saving acceptance probability
        ##self.p = np.ones(self.num_worms)

        dist_ij_shape = np.concatenate( ([self.num_worms, self.num_worms], [len(self.lat_size)]) )
        ##the distance of proposed heads in the next move
        ##for head on collisions
        self.hoc_dist_ij = np.ones(shape=dist_ij_shape, dtype=np.int64)
        ##the distance of proposed heads to old heads in the next move
        ##for lateral collisions
        self.lc_dist_ij = np.ones(shape=dist_ij_shape, dtype=np.int64)

        ##array which keeps track of collisions, arranging all num_worms in collision "classes"
        ##these have to be sampled successively
        self.collisions = np.full(shape=(self.num_worms//2,self.dim*2), fill_value=self.num_worms, dtype=np.int64)
        ##save additional 1 dim list with all worms that participate in collisions
        self.collision_worms = []
        ##bool value which keeps track whether lateral [0] or head on [1] collisions have happened
        self.lc_hoc_bool = np.array([False, False], dtype=bool)

        ##keeps track of heads approaching tails
        ##needed for sampling p correctly using df0
        ##the default is no "catch" (catch_tail=num_worms (invalid index))
        self.catch_tail = np.full(shape=self.num_worms, fill_value=self.num_worms, dtype=np.int64)
        print(f"initializing catch_tail: {self.catch_tail}")

        ##bool value which keeps track whether a worm has closed up
        self.worm_closed = np.full(fill_value = 0, shape=self.num_worms, dtype=bool)

        ##the update of f at the start of the worm should be postponed
        self.starting = np.full(fill_value = 1, shape=self.num_worms, dtype=bool)

        ##also save all the worms which have not collided
        ##these can be sampled independently from the rest
        self.non_collision_worms = []


        print(f"Setting worm to tail: {self.tail}, head: {self.head}\n")
        print(f"with k_delta: {self.k_delta}\n")

        """MCMC variables"""

        self.w_sites = []
        self.b_sites = []
        coords = [list(range(lat_len)) for lat_len in self.lat_size]

        for coord in product(*coords):
            if np.mod(np.sum(coord),2) == 0:
                self.w_sites.append(coord)
            else:
                self.b_sites.append(coord)

        self.w_sites = np.array(self.w_sites)
        self.b_sites = np.array(self.b_sites)

        self.n_w_sites = len(self.w_sites)
        self.n_b_sites = len(self.b_sites)

        #print(f"white sites: {self.n_w_sites}")
        #print(f"black sites: {self.n_b_sites}")

        self.w_moves = np.zeros((self.n_w_sites, self.dim), dtype = int)
        self.b_moves = np.zeros((self.n_b_sites, self.dim), dtype = int)

        self.w_target_sites = np.zeros(shape=self.w_sites.shape, dtype=np.int64)
        self.b_target_sites = np.zeros(shape=self.b_sites.shape, dtype=np.int64)

        self.mcmc_w_mod_dim = np.zeros(shape=self.n_w_sites, dtype = int)

        self.mcmc_b_mod_dim = np.zeros(shape=self.n_w_sites, dtype = int)

        self.w_link_coord = np.zeros((self.n_w_sites, self.dim+1), dtype = int)
        self.b_link_coord = np.zeros((self.n_b_sites, self.dim+1), dtype = int)

        self.mcmc_w_dl = np.zeros(self.n_w_sites, dtype = int)
        self.mcmc_w_df = np.zeros(self.n_w_sites, dtype = int)
        self.mcmc_w_target_df =np.zeros(self.n_w_sites, dtype = int)

        self.mcmc_b_dl = np.zeros(self.n_b_sites, dtype = int)
        self.mcmc_b_df = np.zeros(self.n_b_sites, dtype = int)
        self.mcmc_b_target_df =np.zeros(self.n_b_sites, dtype = int)



    def remove_closed_worms(self, closed_i_worms, open_i_worms, num_worms, head, tail, k_delta, df0, worm_closed, catch_tail, starting, dim):
        """
        If worm simulation needs to be completed, worms have to be remove ocne they are finished
        At the end of a simulation all worms should be closed
        This saves the characteristic attributes of open worms and discards closed onws
        """

        new_num_worms = len(open_i_worms)

        open_worms_head = np.zeros((new_num_worms, 2, dim), dtype = int)
        open_worms_tail = np.zeros((new_num_worms, dim), dtype = int)
        open_worms_k_delta = np.zeros(new_num_worms, dtype = int)
        open_worms_df0 = np.zeros(new_num_worms, dtype = int)
        open_worms_catch_tail = np.full(shape=new_num_worms, fill_value=new_num_worms, dtype=np.int64)

        ##create a filtered copy of old arrays
        for new_i_worm, open_i_worm in enumerate(open_i_worms):
            open_worms_head[new_i_worm] = head[open_i_worm]
            open_worms_tail[new_i_worm] = tail[open_i_worm]
            open_worms_k_delta[new_i_worm] = k_delta[open_i_worm]
            open_worms_df0[new_i_worm] = df0[open_i_worm]

        ##overwrite worm values with filtered versions
        """correct referencing to class attribute?"""
        num_worms = new_num_worms

        head = open_worms_head
        tail = open_worms_tail
        k_delta = open_worms_k_delta
        df0 = open_worms_df0
        catch_tail = open_worms_catch_tail
        """correct referencing to class attribute?"""
        worm_closing = np.full(fill_value = 0, shape=num_worms, dtype=bool)
        starting = np.full(fill_value = 0, shape=num_worms, dtype=bool)




    """Lattice functions"""
    #@jit(nopython=True)
    def read_lat(self, l_links, k_links, read_path):
        """
        read the lattice conf from a file
        """
        lat_links = np.fromfile(read_path, dtype=np.int64, sep=" ").reshape(shape=conf_size)
        l_links = lat_links[0]
        k_links = lat_links[1]
        #l_links = np.fromfile(read_path, dtype=np.int64, sep=" ").reshape(shape=conf_size)

    #@jit(nopython=True)
    def save_lat(self, links, save_path):
        """
        save the lattice conf to a file
        """
        links.flatten().tofile(save_path, sep=" ")


    """Parallelize completely"""
    #@jit(nopython=True)
    def transform_link_index_worm(self, link_coord, move_sign, move, lat_size, dim):
        """
        transforms a lattice index to a valid value.
        the first dimension of link_coord is always a dummy index per default and is ignored in this function
        Needed for links in the negative direction sign=1
        because of the special way configurations are saved
        not all links from a particular site can be accessed by index [lattice_coord, link_index]
        """

        #move_i = link_coord[0] + sign*dim
        #print("transforming link index")
        #print(f"possibly invalid link index:\n {link_coord}")

        """what is faster?"""

        #if sign == 1:
        #    link_coord[1:] += move_set[move_i]
        ##for multiple worms
        link_coord[:,1:] += move_sign * move
        #link_coord[1:] = link_coord[1:] + (sign * move_set[move_i])

        ##impose periodic boundary conditions
        #self.per_lat_coord(link_coord[1:], lat_size, dim)
        ##for multiple worms
        self.per_lat_coord(link_coord[:,1:], lat_size, dim)

        #link_index = [dir_i] + new_lat_coord
        #print(f"valid link index:\n {link_coord}")
        #return link_index

    """Parallelize completely"""
    #@jit(nopython=True)
    def per_lat_coord(self, lat_coord, lat_size, dim):
        """
        impose periodic boundary conditions on the lattice
        the first dimension of link_coord is always a dummy index per default and is ignored in this function
        lat_coord that lie outside of lat size
        have to be glued to the other lattice end
        """
        #print("imposing periodic bc")
        #print(f"lat_coord before bc: \n {lat_coord}")

        #for d in np.arange(start=0, stop=dim, step=1, dtype=np.int64):
        for d in range(0, int(dim), 1):
        #for d in range(0, len(lat_size), 1):
            #if lat_coord is outside of the possible index domain

            #if lat_coord[d] >= lat_size[d]:
            #    lat_coord[d] = (lat_coord[d] - lat_size[d])
            #elif lat_coord[d] < 0:
            #    lat_coord[d] = (lat_coord[d] + lat_size[d])

            ##for multiple worms
            for i in range(len(lat_coord[:,d])):
                if lat_coord[i,d] >= lat_size[d]:
                    lat_coord[i,d] = (lat_coord[i,d] - lat_size[d])
                elif lat_coord[i,d] < 0:
                    lat_coord[i,d] = (lat_coord[i,d] + lat_size[d])

        #print(f"lat_coord after bc: \n {lat_coord}")

        #return lat_coord

    """Parallelize completely"""
    #@jit(nopython=True)
    def get_prob_k_all(self, l_links, k_links, mu, head, tail, worm_link_coord, dk, df, df_new_head, W, f, num_worms, p):
        """
        Calculate the aceptance probability for a worm move
        head is the coordinate vector of the worms heads
        tail is the coordinate vectore of the worms tails
        sign is the orientation of that move: 0=positive, 1=negative
        worm_link_coord is the link which the worm is trying to modify
        dk, df are thes changes it is proposing
        W the physical weights as one long vector
        f as an array of lattice size
        num_worms is the total number of worms for which the probability needs to be calculated
        p is overwritten by this function
        """

        #print(f"Calculating acceptance probability for all {num_worms} worms")

        """overwritting reference or assigning variable?"""
        p = np.ones(shape=num_worms, dtype=np.int64)

        old_head = head[:,0]
        prop_head = head[:,1]

        k_old_link = np.zeros(shape=num_worms, dtype=np.int64)
        k_proposed_link = np.zeros(shape=num_worms, dtype=np.int64)
        l_old_link = np.zeros(shape=num_worms, dtype=np.int64)

        f_old = np.zeros(shape=num_worms, dtype=np.int64)
        f_prop_head = np.zeros(shape=num_worms, dtype=np.int64)


        for i_worm in range(num_worms):
            ##just some values needed for acceptance probability
            k_old_link[i_worm] = k_links[tuple(worm_link_coord[i_worm])]
            k_proposed_link[i_worm] = k_links[tuple(worm_link_coord[i_worm])] + dk[i_worm]
            l_old_link[i_worm] = l_links[tuple(worm_link_coord[i_worm])]

            f_old[i_worm] = f[tuple(head[i_worm,0])]
            f_prop_head[i_worm] = f[tuple(head[i_worm,1])]

            ##different factor in acceptance probability for changing modulus of k link
            ##accounts for factorial coefficient in formula
            if abs(k_proposed_link[i_worm]) > abs(k_old_link[i_worm]):
                p[i_worm] = p[i_worm] / float((abs(k_proposed_link[i_worm]) + l_old_link[i_worm]))
            else:
                p[i_worm] = p[i_worm] * float((abs(k_old_link[i_worm]) + l_old_link[i_worm]))


            """Find a more efficient way"""
            print(f"f_old[i_worm] {f_old[i_worm]}")
            print(f"f_prop_head[i_worm] {f_prop_head[i_worm]}")
            print(f"df[i_worm] {df[i_worm]}")
            print(f"df_new_head[i_worm] {df_new_head[i_worm]}")
            ##the acceptance probability
            
            print(f"f1: {(f_old[i_worm] + df[i_worm])//2}")
            print(f"f2: {(f_prop_head[i_worm] + df_new_head[i_worm])//2}")
            if not np.all(old_head[i_worm] == tail[i_worm]):
                ##worm has already started
                ##multiply p with W[f_prop/2]
                p[i_worm] *= W[(f_old[i_worm] + df[i_worm])//2]/W[(f_prop_head[i_worm] + df_new_head[i_worm])//2]

             ###p[i_worm] *= ( (1 - int(np.all(old_head[i_worm] == tail[i_worm])))*W[int((f_old[i_worm] + df[i_worm])//2)] + int(np.all(old_head[i_worm] == tail[i_worm])) )/W[int(f_prop_head[i_worm]//2)]

            else:
                ##worm has not yet started
                ##multiply p with W[f[head + move]]
                p[i_worm] *= 1./W[int(f_prop_head[i_worm]/2)]

            ##if direction is timelike (dir_i = 0) multiply by exp(-change*mu)
            if worm_link_coord[i_worm][0] == 0:
                #p *= np.exp((1. - 2.*sign)*mu*value)
                p[i_worm] *= np.exp(float(dk[i_worm])*mu)

        #print(f"k_old {k_old_link}")
        #print(f"dk {dk}")
        #print(f"k_prop {k_proposed_link}")
        #print(f"df {df}")
        #print(f"df_new_head {df_new_head}")
        #print(f"p {p}")



    def get_prob_k_one(self, l_links, k_links, mu, head, tail, worm_link_coord, i_worm, dk, df, df_new_head, W, f, p):
        """
        Calculate the aceptance probability for a worm move
        head is the coordinate vector of the worms heads (all of them!!!)
        tail is the coordinate vectore of the worms tails (all of them!!!)
        worm_link_coord is the link which the worm is trying to modify (all of them!!!)
        dk, df are thes changes it is proposing (for one worm!!!)
        W the physical weights as one long vector
        f as an array of lattice size
        num_worms is the total number of worms for which the probability needs to be calculated
        p is overwritten by this function
        """

        #print(f"Calculating acceptance probability for worm {i_worm}")

        old_head = head[i_worm,0]
        prop_head = head[i_worm,1]


        ##just some values needed for acceptance probability
        #for i_worm in range(num_worms):
        k_old_link = k_links[tuple(worm_link_coord[i_worm])]
        k_proposed_link = k_links[tuple(worm_link_coord[i_worm])] + dk[i_worm]
        l_old_link = l_links[tuple(worm_link_coord[i_worm])]

        f_old = f[tuple(head[i_worm,0])]
        f_prop_head = f[tuple(head[i_worm,1])]



        ##different factor in acceptance probability for changing modulus of k link
        ##accounts for factorial coefficient in formula
        #for i_worm in range(num_worms):
        if abs(k_proposed_link) > abs(k_old_link):
            p = p / (abs(k_proposed_link) + l_old_link)
        else:
            p = p * (abs(k_old_link) + l_old_link)


        """Find a more efficient way"""
        ##for i_worm in range(num_worms):
        if not np.all(old_head == tail):
            ##worm has already started
            ##multiply p with W[f_prop/2]
            """divide by 2"""
            p *= W[int((f_old + df)//2)]/W[int((f_prop_head + df_new_head)//2)]

        else:
            ##worm has not yet started
            ##multiply p with W[f[head + move]]
            p *= 1./W[int((f_prop_head + df_new_head)//2)]

        ###p *= ( (1 - int(np.all(old_head == tail)))*W[int((f_old + df)//2)] + int(np.all(old_head == tail)) )/W[int(f_prop_head//2)]

        ##if direction is timelike (dir_i = 0) multiply by exp(-change*mu)
        if worm_link_coord[i_worm][0] == 0:
            #p *= np.exp((1. - 2.*sign)*mu*value)
            p *= np.exp(float(dk)*mu)

        #print(f"k_old {k_old_link}")
        #print(f"dk {dk}")
        #print(f"k_prop {k_proposed_link}")
        #print(f"df {df}")
        #print(f"df_new_head {df_new_head}")
        #print(f"p {p}")




    """Parallelize partially"""
    def sample_k_worm_collision_queues(collisions, l_links, k_links, mu, head, tail, worm_link_coord, dk, df, W, f, worm_closed, starting, catch_tail, num_worms):

        #print("Starting Metropolis algorithm for collision queues")
        #keep track of which worms moved in the last queue iteration
        #prev_updated_worms = np.full(0, size=len(collisions), dtype=bool)

        #df_prev_queue = np.full(0, size=len(collisions), dtype=np.int64)
        #df_current_queue = np.full(0, size=len(collisions), dtype=np.int64)

        ##DONT parallelize this loop
        for queue_i, collision_worm_queue in enumerate(collisions[:]):

            print(f"queue {queue_i}")
            print(f"worms {collision_worm_queue}")
            ##acceptance probability
            p_queue = np.ones(len(collision_worm_queue), dtype=float)
            ##randomly drawn probability for metropolis algorithm
            p_draw = np.random.uniform(low=0., high= 1., size=len(collision_worm_queue))


            """parallelize this loop?"""
            ##colliding worms that can be safely updated at once
            for col_worm_i, i_worm in enumerate(collision_worm_queue):

                ##queue length can vary
                ##collision_worm_queue is always a square matrix
                ##with fields filled in with num_worms, which should be ignored
                if i_worm != num_worms:

                    ##placeholder for the intrinsic changes
                    df_i = df[i_worm]
                    dk_i = dk[i_worm]

                    ##in case of a head on collision,
                    ##the approaching worm should become aware
                    ##of the change in f at the newly set head
                    ##the f update is done onle at the old head!
                    ##therefore determine whether the current new heads f value has to be changed
                    ##because it has not been udpated yet!
                    df_new_head = 0
                    """ACCOUNT FOR MULTIPLE WORMS COLLIDING IN THE SAME POINT"""
                    if queue_i > 1:
                        prev_worms_in_col_class = collision_worm_queue[col_worm_i][:queue_i]
                        for prev_worm_i in prev_worms_in_col_class:
                            df_new_head += np.all( head[i_worm,1] == head[prev_worm_i,0] )*df[prev_worm_i]

                    """CHECK FOR TAILS???"""
                    ##the same is true for new heads approaching tails
                    ##the unaccounted value df0 has to be taken
                    if catch_tail[i_worm] != num_worms:
                        df_new_head += df[catch_tail[i_worm]]

                    """p_queue reference or assignment?"""
                    self.get_prob_k_one(l_links, k_links, mu, head, tail, worm_link_coord, i_worm, dk_i, df_i, df_new_head, W, f, worm_closed, num_worms, p_queue[col_worm_i])
                    ##Metropolis algorithm
                    if p_draw[col_worm_i] < p_queue[col_worm_i]:
                        ##accept the move

                        ##worm has started if accepted move
                        if starting[i_worm] == 1:
                            starting[i_worm] = 0
                        ##set boolean value to 1
                        ##print("move accepted")
                        #updated_worms[i_worm] = 1

                        ##adapt k and f
                        k[tuple(worm_link_coord[i_worm])] += dk_i
                        f[tuple(head[i_worm,0])] += df_i

                        ##move the head
                        head[i_worm,0] = head[i_worm,1].copy()
                        ###head[i_worm,0] += moves[i_worm]

                        ##check whether head == tail
                        ##and save it
                        worm_closed[i_worm] = np.all( head[i_worm,0] == tail[i_worm] )


                    """circumvent if statement by expression?"""
                    ##adapt k and f
                    ###k[worm_link_coord[i_worm]] += ( float(p_draw < p_queue[col_worm_i]) )*dk[i_worm]
                    ###f[head[i_worm,0]] += ( float(p_draw < p_queue[col_worm_i]) )*df[i_worm]

                    ##move the head
                    ###head[i_worm,0] = ( float(p_draw < p_queue[col_worm_i]) )*head[i_worm,1].copy() + ( float(p_draw >= p_queue[col_worm_i]) )*head[i_worm,0]
                    ###head[i_worm,0] += ( float(p_draw < p_queue[col_worm_i]) )*moves[i_worm]


                    """not necessary if masking is used in regular sampling?"""
                    """careful with head on and tail collisions ..."""
                    ##reset the worms values to default (unchanged and f, no move)
                    head[i_worm,1] = head[i_worm,0].copy()
                    moves[i_worm] = np.array([0,0], dtype=np.int64)
                    #dk[i_worm] = 0
                    #df[i_worm] = 0




    def set_df_dk_zero(df, dk, i_worms):

        for i_worm in i_worms:
            dk[i_worm] = 0
            df[i_worm] = 0

    """Parallelize completely"""
    def sample_k_worm_all(self, l_links, k_links, mu, head, tail, worm_link_coord, dk, df, W, f, worm_closed, starting, catch_tail, num_worms):

            p_all = np.ones(num_worms, dtype=float)
            p_draw = np.random.uniform(low=0., high= 1., size=num_worms)

            df_new_head = np.zeros(num_worms, dtype=np.int64)
            """CHECK FOR TAILS"""
            print(f"catch_tail {catch_tail}")
            for i_worm in range(num_worms):
                ##the same is true for new heads approaching tails
                ##the unaccounted value df0 has to be taken
                if catch_tail[i_worm] != num_worms:
                    print(f"caught tail of {catch_tail[i_worm]}")
                    df_new_head[i_worm] += df[catch_tail[i_worm]]

            #print("Starting Metropolis algorithm for all worms")
            #print(f"dk: {dk}")
            #print(f"df: {df}")
            #print(f"df_new_head: {df_new_head}")
            #( l_links, k_links, mu, head, tail, worm_link_coord, dk, df, df_new_head, W, f, num_worms, p):
            self.get_prob_k_all(l_links, k_links, mu, head, tail, worm_link_coord, dk, df, df_new_head, W, f, num_worms, p_all)
            """Combine both i_worm loops"""
            for i_worm in range(num_worms):
                ###"""p_queue reference or assignment?"""
                ###get_prob_k_one(self, l_links, k_links, mu, head, tail, worm_link_coord, i_worm, dk, df, W, f, p_all[i_worm]):
                if p_draw[i_worm] < p_all[i_worm]:
                    ###accept the move, adapt k and f

                    ##worm has started if accepted move
                    if starting[i_worm] == 1:
                            starting[i_worm] = 0

                    #print("accepted")
                    #print(f"worm {i_worm} adjusts link {worm_link_coord[i_worm]}")
                    """INDEX 0 OUT OF BOUNDS???"""
                    #print(f"k before {k_links[tuple(worm_link_coord[i_worm])]}")
                    k_links[tuple(worm_link_coord[i_worm])] += dk[i_worm]
                    #print(f"k after {k_links[tuple(worm_link_coord[i_worm])]}")
                    #print(f"f before {f[tuple(head[i_worm,0])]}")
                    f[tuple(head[i_worm,0])] += df[i_worm]
                    #print(f"f after {f[tuple(head[i_worm,0])]}")
                    #k_change[worm_link_coord[i_worm]] += dk[i_worm]
                    #f_change[head[i_worm,0]] += df[i_worm]

                    ##move the head
                    #print(f"head before {head[i_worm,0]}")
                    head[i_worm,0] = head[i_worm,1].copy()
                    #print(f"head after {head[i_worm,0]}")

                    ##check whether head == tail
                    ##and save it
                    worm_closed[i_worm] = np.all( head[i_worm,0] == tail[i_worm] )

                """circumvent if statement by expression?"""
                ##adapt k and f
                ###k[worm_link_coord[i_worm]] += ( float(p_draw < p_queue[col_worm_i]) )*dk[i_worm]
                ###f[head[i_worm,0]] += ( float(p_draw < p_queue[col_worm_i]) )*df[i_worm]

                ##move the head
                ###head[i_worm,0] = ( float(p_draw < p_queue[col_worm_i]) )*head[i_worm,1].copy() + ( float(p_draw >= p_queue[col_worm_i]) )*head[i_worm,0]
                ###head[i_worm,0] += ( float(p_draw < p_queue[col_worm_i]) )*moves[i_worm]

                """not necessary if masking is used in regular sampling?"""
                """careful with head on and tail collisions ..."""
                head[i_worm,1] = head[i_worm,0].copy()
                #moves[i_worm] = np.array([0,0], dtype=np.int64)


    #@jit(nopython=True)
    def print_env_worm(self, lat_size, head, tail):
        """
        simple function for plotting worms head and tail
        for visualization
        """

        """
        l_links_host = l_links.copy_to_host()
        k_links_host = k_links.copy_to_host()
        head_host = head.copy_to_host()
        tail_host = tail.copy_to_host()
        """
        image = np.zeros(lat_size)
        #image = np.zeros(k_links_host[0].shape)
        head_val = 10
        tail_val = -10
        #print(f"printing tail: {tail}")
        #print(f"printing head: {head}")
        image[tuple(head)] = head_val
        image[tuple(tail)] = tail_val

        #print(image)
        plt.imshow(image)
        plt.show()


    """redundant?"""
    def check_head_on_collision(self, head, hoc_dist_ij, bool_hoc):
        """
        Check whether worms in the simulation are going to have a head on collision in the next move
        """

        for i_worm in range(self.n_worms):
            for j_worm in range(i_worm+1,self.n_worms):
                dist = head[i_worm,1] - head[j_worm,1]
                hoc_dist_ij[i_worm,jworm] = dist

                if np.all(dist == 0):
                    ##collision between i_worm and j_worm
                    bool_hoc = True


    """redundant?"""
    def check_lateral_collision(self, head, lc_dist_ij, bool_lc):
        """
        Check whether worms in the simulation are going to have a lateral collision in the next move
        """

        for i_worm in range(self.n_worms):
            for j_worm in range(i_worm+1,self.n_worms):
                dist = head[i_worm,1] - head[j_worm,0]
                lc_dist_ij[i_worm,jworm] = dist

                if np.all(dist == 0):
                    bool_lc = True


    def count_collisions(self, head, tail, hoc_dist_ij, lc_dist_ij, num_worms, np_collisions, collision_worms, non_collision_worms, lc_hoc_bool, catch_tail):
        """
        Count colliding worms in the simulation and separate them into collision "classes"
        can not be paralellized as the counting/separation procedure depends on the previous record
        in addition head and tail collisions are counted
        these however dont pose a problem when sampling in parallel
        """
        collisions = []

        ##keep track of which worm belongs to which collision
        ##per default they are part of no collision (index=num_worms)
        worm_col_i = np.full(fill_value=num_worms, shape=num_worms,dtype=np.int64)

        lc_hoc_bool[0] = False
        lc_hoc_bool[1] = False

        ###skippable_worms = np.full(num_worms, fill_value=num_worms, dtype=np.int64)

        ##count all collisions
        col_counter = 0

        print("counting collisions")
        ##iterate through all worms
        for i_worm in range(num_worms):

            ##check whether the current worm is part of an earlier collision class
            if worm_col_i[i_worm] == num_worms:
                print(f"creating new collision for {i_worm}")
                ##if not, then create a new collision class with index col_counter
                ##later it will be clear whether this worm really takes part in a collision
                ##in which case i_worm_collision will be appended to collisions
                old_collision = False
                col_i = col_counter
                #collisions.append([i_worm])
                #i_worm_collision = collisions[-1]
                i_worm_collision = [i_worm]
            else:
                ##if so access the old collision array with index col_i[i_worm]
                old_collision = True
                col_i = worm_col_i[i_worm]
                i_worm_collision = collisions[col_i]
                print(f"accessing old collision {col_i} for {i_worm}")
                print(i_worm_collision)

            ##range starts at i_worm+1 because distance measure is symmetric
            for j_worm in range(i_worm+1,num_worms):

                ##calculate the distance between worms (heads)
                ##both head on and lateral collisions (hoc and ld)
                hoc_dist = head[i_worm,1] - head[j_worm,1]
                lc_dist = head[i_worm,1] - head[j_worm,0]
                hoc_dist_ij[i_worm,j_worm] = hoc_dist
                lc_dist_ij[i_worm,j_worm] = lc_dist

                ##calculate the distnace between new heads and tails
                i_head_j_tail_dist = head[i_worm,1] - tail[j_worm]
                j_head_i_tail_dist = head[j_worm,1] - tail[i_worm]

                ##remember the possible tail a worm is about to move to
                ##for sampling p correctly with df0 corresponding to that tail
                ##the default is no "catch" (catch_tail=num_worms (invalid index))
                if np.all(i_head_j_tail_dist == 0):
                    print(f"{i_worm} catches {j_worm}")
                    catch_tail[i_worm] = j_worm
                if np.all(j_head_i_tail_dist == 0):
                    print(f"{i_worm} catches {j_worm}")
                    catch_tail[j_worm] = i_worm

                ##check whether i_worm and j_worm collide
                if np.all(hoc_dist == 0) or np.all(lc_dist == 0):

                    ##if so then set the boolean to True once in for all
                    if np.all(hoc_dist == 0):
                        print(f"head on collision between {i_worm} and {j_worm}")
                        lc_hoc_bool[0] = True
                    if np.all(lc_dist == 0):
                        print(f"lateral collision between {i_worm} and {j_worm}")
                        lc_hoc_bool[1] = True

                    ##set the collision index of the new worm appropriately
                    worm_col_i[j_worm] = col_i

                    """reference or copy of collisions"""
                    i_worm_collision.append(j_worm)

            ##do some extra things if a new collision (i_worm_collision) was recorded, but not yet appended to collisions
            if (old_collision == False) and (len(i_worm_collision) >= 2):

                ##this list contains a real collision with 1+ partners
                ##increase the collision counter
                col_counter += 1
                ##set the collision index of this worm to the new collision
                worm_col_i[i_worm] = col_i
                ##and add it to all collisions
                collisions.append(i_worm_collision)

        ###collision_worms = np.flatten(collisions, dtype=np.int64)
        collision_worms = [col_i_worm for collision in collisions for col_i_worm in collision]
        non_collision_worms = [i for i in range(num_worms) if i not in collision_worms]

        print(f"raw collisions:")
        print(collisions)
        """MAKE COLLISIONS NON RECTANGULAR"""
        ##collisions now is a regular non-rectangular python list
        ##cast it to a rectangular numpy array
        ##fill empty values with value of num_worms
        for collision in collisions:
            num_fill = num_worms//2 - len(collision) + 1
            print(f"num_fill {num_fill}")
            if num_fill > 0:
                collision.extend( num_fill*[num_worms])



        print(f"rectangular collisions:")
        print(collisions)

        np_collisions = np.array(collisions, dtype=np.int64)


    """Worm functions"""


    #@jit(nopython=True)
    def propose_moves(self, move_dim, move_sign, move_i, move_set, moves, head, worm_link_coord, k_delta, k_links, dk, df, k_change, f_change, catch_tail, num_worms, lat_size, dim):
        """
        Draws a random move from the moveset
        as well as a random value for changing the link value
        and prepares some values needed lateron for the probabalistic sampling step
        """

        ##reset k and f change when proposing new moves for all worms
        df.fill(0)
        dk.fill(0)
        k_change.fill(0)
        f_change.fill(0)
        catch_tail.fill(num_worms)


        ##draw a ranom move dimension (dir_i) and orientation (sign)
        #dir_i = np.random.randint(low=0,high=dim)
        move_dim = np.random.randint(low=0, high=dim, size=num_worms)
        #print(f"proposing move_dim:\n {move_dim}")
        #sign = np.random.randint(low=0, high=2)
        move_sign = np.random.randint(low=0, high=2, size=num_worms)
        #print(f"proposing sign:\n {move_sign}")

        ##calculate the corresponding move_index for the move_set
        move_i = move_dim + (dim * move_sign)
        #print(f"proposing move_i:\n {move_i}")

        ##and move
        for i_worm in range(num_worms):
            moves[i_worm] = move_set[ move_i[i_worm] ]

        #print(f"proposing moves:\n {moves}")

        ##the link value which is to be modified if the move is accepted
        #worm_link_coord[0] = dir_i
        worm_link_coord[:,0] = move_dim
        worm_link_coord[:,1:] = head[:,0].copy()
        ##check whether the link index has to be changed in order to be accessed
        self.transform_link_index_worm(worm_link_coord, move_sign, moves, lat_size, dim)

        ##the new proposed head if the move is accepted
        #prop_head = head + move_set[move_i]
        #prop_head += move_set[move_i]
        """RESET PROP HEAD ALWAYS TO HEAD"""
        ###head[:,1] += moves
        head[:,1] = head[:,0] + moves
        ###for i_worm in range(num_worms):
        ###    head[i_worm,1] = head[i_worm,0] + moves[i_worm]

        #print(f"proposing new heads:\n {head[:,1]}")

        ##check whether the new head has to be changed for periodic bc
        self.per_lat_coord(head[:,1], lat_size, dim)

        #print(f"proposing new heads:\n {head[:,1]}")

        for i_worm in range(num_worms):
            old_k = k_links[tuple(worm_link_coord[i_worm])]

            """NOT CORRECTLY SETTING DF"""
            ##changes whic hshould be kept track of for each worm
            dk_p = (1-2*move_sign[i_worm]) * k_delta[i_worm]
            #if np.sign(dk_p) == np.sign(old_k):
            df_p = 2*(abs(old_k+dk_p) - abs(old_k))

            #if abs(old_k+dk_p) > abs(old_k):
            #    df_p = 2*dk_p.copy()
            #else:
            #    df_p = -2*dk_p.copy()

            dk[i_worm] = dk_p
            df[i_worm] = df_p
        
        print(f"k_old: {old_k}")
        print(f"dk: {dk}")
        print(f"df: {df}")

        ##calculate the k_change array which is then globally added to the lattice
        ##when doing regular independent sampling of all worms (no collisions!)
        #for i_worm in range(self.num_worms):
        #    k_change[worm_link_coord[i_worm]] += dk[i_worm]
        #    f_change[head[i_worm,0]] += df[i_worm]




    #@jit(nopython=True)
    def reset_worm(self, tail, head, worm_link_coord, i_worm, k_delta, lat_size, worm_closed, starting, dim):
        """
        reset the worm randomly
        choose the value of k_delta (fixed!)
        """
        for d in range(dim):
            tail[i_worm,d] = np.random.randint(low=0, high=lat_size[d])
            head[i_worm,0,d] = tail[i_worm,d]
            worm_link_coord[i_worm,d+1] = head[i_worm,0,d]
        worm_link_coord[i_worm,0] = 0
        k_delta[i_worm] = np.random.randint(low=0, high=2)*2 - 1
        worm_closed[i_worm] = 0
        starting[i_worm] = 1
        print(f"Resetting worm {i_worm} to tail: {tail[i_worm]}, head: {head[i_worm,0]}\n")
        print(f"with k_delta: {k_delta[i_worm]}\n")



    """Monte Carlo functions"""

    def sweep(self, l_links, k_links, w_sites, b_sites, move_set, W, f, n_w_sites, n_b_sites, dim):
        """
        goal: do a parallel sweep through all given lattice sites
        """

        #mc_sites(sites, mod_dim, dl, df, df_target, l_links, k_links, W, f, n_sites, move_set, dim):
        self.mc_sites(w_sites, l_links, k_links, W, f, n_w_sites, move_set, dim)
        self.mc_sites(b_sites, l_links, k_links, W, f, n_b_sites, move_set, dim)


    def draw_site(self, i_site, mod_dim, dl, df, df_target, link_coord, sites, target_sites, l_links, move_set, moves, n_sites, lat_size, dim):
        """
        choose random modification at random link dimension
        """
        """find better method"""
        mod_dim[i_site] = np.random.randint(dim)
        if l_links[tuple(link_coord[i_site])] == 0:
            dl[i_site] = 1
        else:
            dl[i_site] = (2*np.random.randint(low=0, high=2) - 1)
        
        df[i_site] = 2*dl[i_site]
        df_target[i_site] = -df[i_site]
        link_coord[i_site,0] = mod_dim[i_site]
        link_coord[i_site,1:] = sites[i_site]
        
        moves[i_site] = move_set[mod_dim[i_site]]    


    def draw_sites(self, mod_dim, dl, df, df_target, link_coord, sites, target_sites, l_links, move_set, moves, n_sites, lat_size, dim) :
        """
        choose random modifications at random link dimensions
        """
        mod_dim = np.random.randint(low=0, high=dim, size=n_sites)
        dl[:] = (2*np.random.randint(low=0, high=2, size=n_sites) - 1)[:]

        for i_site in range(n_sites):
            if l_links[tuple(link_coord[i_site])] == 0:
                dl[i_site] = 1
            moves[i_site] = move_set[mod_dim[i_site]]
            
        #print(f"MCMC dl:\n{dl}")
        """CHECK WHETHER CORRECT"""
        df[:] = 2*dl[:]
        df_target[:] = -df[:]
        link_coord[:,0] = mod_dim.copy()
        link_coord[:,1:] = sites.copy()

        target_sites[:] = (sites + moves)[:]

        #(lat_coord, lat_size, dim):
        self.per_lat_coord(target_sites, lat_size, dim)

        #print(f"MCMC link_coord:\n{link_coord}")
        #print(f"MCMC target site:\n{target_sites}")

    def mc_sites(self, sites, target_sites, link_coord, mod_dim, dl, df, df_target, l_links, k_links, W, f, n_sites, move_set, moves, lat_size, dim):

        """
        MCMC for multiple lattice sites
        """

        p_draw = np.random.uniform(low=0., high=1., size=n_sites)
        p = np.ones(len(sites))

        ##draw link and link_modification at random
        self.draw_sites(mod_dim, dl, df, df_target, link_coord, sites, target_sites, move_set, moves, n_sites, lat_size, dim)

        """parallelize this loop"""
        for i_site in range(len(sites)):

            ##get acceptance probability
            self.get_prob_l_site(l_links, k_links, sites, link_coord, target_sites, i_site, dl, df, df_target, W, f, p)

            #if p_draw[i_site] < p[i_site]:
            #if p_draw[i_site] < 0.5:
            if True:
                #accept
                #print("accepted")
                l_links[tuple(link_coord[i_site])] += dl[i_site]
                f[tuple(sites[i_site])] += df[i_site]
                f[tuple(target_sites[i_site])] += df_target[i_site]
            else:
                #print("rejected")
                pass


    def mc_site(self, i_sites, sites, target_sites, link_coord, mod_dim, dl, df, df_target, l_links, k_links, W, f, n_sites, move_set, moves, lat_size, dim):

        """
        MCMC for one single lattice site
        """

        p_draw = np.random.uniform(low=0., high=1.)
        p = np.ones(1)

        ##draw link and link_modification at random
        self.draw_site(i_site, mod_dim, dl, df, df_target, link_coord, sites, target_sites, move_set, moves, n_sites, lat_size, dim)


        ##get acceptance probability
        self.get_prob_l_site(l_links, k_links, sites, link_coord, target_sites, i_site, dl, df, df_target, W, f, p)

        #if p_draw[i_site] < p[i_site]:
        #if p_draw[i_site] < 0.5:
        if True:
            #accept
            #print("accepted")
            l_links[tuple(link_coord[i_site])] += dl[i_site]
            f[tuple(sites[i_site])] += df[i_site]
            f[tuple(target_sites[i_site])] += df_target[i_site]
        else:
            #print("rejected")
            pass


    def get_prob_l_site(self, l_links, k_links, sites, link_coord, target_site, i_site, dl, df, df_target, W, f, p):
        p[i_site] = 1.0

        #print(f"f_old at site {sites[i_site]}")
        #print(f"f_target at site {target_site[i_site]}")
        f_old = f[tuple(sites[i_site])]
        f_target = f[tuple(target_site[i_site])]

        l_old_link = l_links[tuple(link_coord[i_site])]
        l_proposed_link = l_links[tuple(link_coord[i_site])] + dl[i_site]
        k_old_link = k_links[tuple(link_coord[i_site])]
        #print(f"l_old {l_old_link}")
        #print(f"l_prop {l_proposed_link}")


        #different factor in acceptance probability for changing modulus of l link
        #accounts for factorial coefficient
        if abs(l_proposed_link) > abs(l_old_link):
            p[i_site] = p[i_site]/((abs(k_old_link) + l_proposed_link) * l_proposed_link)
        else:
            p[i_site] = p[i_site]*(abs(k_old_link) + l_old_link) * l_old_link

        p[i_site] *= W[int((f_old + df[i_site])/2)]/W[int(f_target/2)]

 