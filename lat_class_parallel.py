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


        ##size of one link field on the lattice
        self.link_size = np.concatenate(([len(self.lat_size)], self.lat_size))
        
        ##size of one lattice conf
        conf_dof = 2
        self.conf_size = np.concatenate(([conf_dof], [len(self.lat_size)], self.lat_size))
        print(self.conf_size)
        
        ##initialize with zeros
        self.lat_links = np.zeros(shape=self.conf_size, dtype=int)
        self.l_links = np.zeros(shape=self.link_size, dtype=int)
        self.k_links = np.zeros(shape=self.link_size, dtype=int)

        ##an array saving the current status of the f function
        self.f = np.zeros(shape=self.lat_size, dtype=int)
        
        ##the weights whic hare needed for calculating probabilities
        #W = np.fromfile("weights.dat", sep=",")
        """Read weights from file"""
        self.W = np.ones(10000)
        
        
        """Worm variables"""
        
        self.num_worms = 2
        
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
        
    
        """
        f_gpu = cuda.to_device(f)
        w_gpu = cuda.to_device(w)
        """
        
        
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
        self.p = np.ones(self.num_worms)
        
        dist_ij_shape = np.concatenate( ([self.num_worms, self.num_worms], [len(self.lat_size)]) )
        ##the distance of proposed heads in the next move
        ##for head on collisions
        self.hoc_dist_ij = np.ones(shape=dist_ij_shape, dtype=int)
        ##the distance of proposed heads to old heads in the next move
        ##for lateral collisions
        self.lc_dist_ij = np.ones(shape=dist_ij_shape, dtype=int)
        
        ##array which keeps track of collisions, arranging all num_worms in collision "classes"
        ##these have to be sampled successively
        self.collisions = np.full(shape=(self.num_worms//2,self.dim*2), fill_value=self.num_worms, dtype=int)
        ##save additional 1 dim list with all worms that participate in collisions
        self.collision_worms = []
        ##bool value which keeps track whether lateral [0] or head on [1] collisions have happened
        self.lc_hoc_bool = np.array([False, False], dtype=bool)
        
        ##bool value which keeps track whether a worm has closed up
        self.worm_closed = np.full(fill_value = 0, shape=self.num_worms, dtype=bool)
        
        ##also save all the worms which have not collided
        ##these can be sampled independently from the rest
        self.non_collision_worms = []
        

        print(f"Setting worm to tail: {self.tail}, head: {self.head}\n")
        print(f"with k_delta: {self.k_delta}\n")

    """Lattice functions"""
    #@jit(nopython=True)
    def read_lat(self, l_links, k_links, read_path):
        """
        read the lattice conf from a file
        """
        lat_links = np.fromfile(read_path, dtype=int, sep=" ").reshape(shape=conf_size)
        l_links = lat_links[0]
        k_links = lat_links[1]
        #l_links = np.fromfile(read_path, dtype=int, sep=" ").reshape(shape=conf_size)

    #@jit(nopython=True)
    def save_lat(self, links, save_path):
        """
        save the lattice conf to a file
        """
        links.flatten().tofile(save_path, sep=" ")



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
        print("transforming link index")
        print(f"possibly invalid link index:\n {link_coord}")

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
        print(f"valid link index:\n {link_coord}")
        #return link_index

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
        
        #for d in np.arange(start=0, stop=dim, step=1, dtype=int):
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

    """redundant?"""
    #@jit(nopython=True)
    #def update_link_worm(links, lat_size, dim, lat_coord, dir_i, sign, move_set, value):
    def update_link(self, links, link_coord, value):
        """
        update links[links_coord] by addig value 
        """

        links[tuple(link_coord)] += value
        print(f"updating links[{link_coord}] += {value}")
    
    
    """redundant?"""
    def update_links_change(self, links, change):
        """
        update links[links_coord] by addig value 
        """

        links += change
        print(f"updating links with change")

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
        
        print(f"Calculating acceptance probability for all {num_worms} worms")
        
        """overwritting reference or assigning variable?"""
        p = np.ones(shape=num_worms, dtype=int)
        
        old_head = head[:,0]
        prop_head = head[:,1]

        k_old_link = np.zeros(shape=num_worms, dtype=int)
        k_proposed_link = np.zeros(shape=num_worms, dtype=int)
        l_old_link = np.zeros(shape=num_worms, dtype=int)
        
        f_old = np.zeros(shape=num_worms, dtype=int)
        f_prop_head = np.zeros(shape=num_worms, dtype=int)
        
        
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
            ##the acceptance probability 
            if not np.all(old_head[i_worm] == tail[i_worm]):
                ##worm has already started
                ##multiply p with W[f_prop/2]
                p[i_worm] *= W[int((f_old[i_worm] + df[i_worm])//2)]/W[int( (f_prop_head[i_worm]+ df_new_head[i_worm])//2)]
                
             ###p[i_worm] *= ( (1 - int(np.all(old_head[i_worm] == tail[i_worm])))*W[int((f_old[i_worm] + df[i_worm])//2)] + int(np.all(old_head[i_worm] == tail[i_worm])) )/W[int(f_prop_head[i_worm]//2)]

            else:
                ##worm has not yet started
                ##multiply p with W[f[head + move]]
                p[i_worm] *= 1./W[int(f_prop_head[i_worm]/2)]

            ##if direction is timelike (dir_i = 0) multiply by exp(-change*mu)
            if worm_link_coord[i_worm][0] == 0:
                #p *= np.exp((1. - 2.*sign)*mu*value)
                p[i_worm] *= np.exp(float(dk[i_worm])*mu)

        print(f"k_old {k_old_link}")
        print(f"dk {dk}")
        print(f"k_prop {k_proposed_link}")
        print(f"df {df}")
        print(f"df_new_head {df_new_head}")
        print(f"p {p}")
    
             
                
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
        
        print(f"Calculating acceptance probability for worm {i_worm}")
        
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
            
        print(f"k_old {k_old_link}")
        print(f"dk {dk}")
        print(f"k_prop {k_proposed_link}")
        print(f"df {df}")
        print(f"df_new_head {df_new_head}")
        print(f"p {p}")
    
    """redundant?"""
    def get_prob_k_queue(self, l_links, k_links, mu, head, tail, worm_link_coord, i_worm, dk, df, W, f, worm_queue, p_queue):
        
        for queue_i, i_worm in enumerate(worm_queue):
            self.get_prob_k_one(l_links, k_links, mu, head, tail, worm_link_coord, i_worm, dk, df, df_new_head, W, f, p_queue[queue_i])
            
        
    
    def sample_k_worm_collision_queues(collisions, l_links, k_links, mu, head, tail, worm_link_coord, dk, df, W, f, worm_closed, num_worms):
        
        print("Starting Metropolis algorithm for collision queues")
        #keep track of which worms moved in the last queue iteration
        #prev_updated_worms = np.full(0, size=len(collisions), dtype=bool)
        
        #df_prev_queue = np.full(0, size=len(collisions), dtype=int)
        #df_current_queue = np.full(0, size=len(collisions), dtype=int)
        
        ##DONT parallelize this loop
        for queue_i, collision_worm_queue in enumerate(collisions[:]):
            
            print(f"queue {queue_i}")
            print(f"worms {collision_worm_queue}")
            ##acceptance probability
            p_queue = np.ones(len(collision_worm_queue), dtype=float)
            ##randomly drawn probability for metropolis algorithm
            p_draw = np.random.uniform(low=0., high= 1., size=len(collision_worm_queue))
            
            
            ##parallelize this loop
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
                    ###for updated_i_worm in range(len(collisions[:]):
                    ##accomodate for not directly updated f value
                    ###if prev_updated_worms[updated_i_worm]  and np.all(head[i_worm,1] == head[updated_i_worm,0]):
                    ###    df_new_head += df[updated_i_worm]
                                                
                    ##df at new head from worm that came in previous iteration
                    ##a missing update in the f value, due to the way the f update is applied at each old head (not new!)
                    if queue_i > 1:
                        prev_worms_in_col_class = collision_worm_queue[col_worm_i][:queue_i]
                        for prev_worm_i in prev_worms_in_col_class:
                            df_new_head += np.all( head[i_worm,1] == head[prev_worm_i,0] )*df[prev_worm_i]
                    ###if queue_i > 1:                            
                    ###    df_new_head += ( np.all( head[i_worm,1] == head[prev_worm_queue[col_worm_i]] ) )* prev_df[col_worm_i]
                                                
                    """CHECK FOR TAILS???"""

                    """p_queue reference or assignment?"""
                    self.get_prob_k_one(l_links, k_links, mu, head, tail, worm_link_coord, i_worm, dk_i, df_i, df_new_head, W, f, worm_closed, num_worms, p_queue[col_worm_i])
                    ##Metropolis algorithm
                    if p_draw[col_worm_i] < p_queue[col_worm_i]:
                        ##accept the move
                        
                        ##set boolean value to 1
                        ##print("move accepted")
                        #updated_worms[i_worm] = 1
                        
                        ##adapt k and f
                        ###k[worm_link_coord[i_worm]] += dk[i_worm]
                        ###f[head[i_worm,0]] += df[i_worm]
                        
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

                        
                    ##keep track of which worms moved in the last queue iteration
                    #prev_updated_worms = updated_worms.copy()
                    #prev_df[col_worm_i] = df_i.copy()

                    """not necessary if masking is used in regular sampling?"""
                    """careful with head on and tail collisions ..."""
                    ##reset the worms values to default (unchanged and f, no move)
                    head[i_worm,1] = head[i_worm,0].copy()
                    moves[i_worm] = np.array([0,0], dtype=int)
                    #dk[i_worm] = 0
                    #df[i_worm] = 0
            
            ##remembering the worm_queue for the next run
            #prev_worm_queue = collision_worm_queue.copy()
            
            
    
    def set_df_dk_zero(df, dk, i_worms):
        
        for i_worm in i_worms:
            dk[i_worm] = 0
            df[i_worm] = 0
                                                
                                                
    def sample_k_worm_all(self, l_links, k_links, mu, head, tail, worm_link_coord, dk, df, W, f, worm_closed, num_worms):
            
            p_all = np.ones(num_worms, dtype=float)
            p_draw = np.random.uniform(low=0., high= 1., size=num_worms)
            
            df_new_head = np.zeros(num_worms, dtype=int)
            """CHECK FOR TAILS"""
            
            
            print("Starting Metropolis algorithm for all worms")
            print(f"dk: {dk}")
            print(f"df: {df}")
            #( l_links, k_links, mu, head, tail, worm_link_coord, dk, df, df_new_head, W, f, num_worms, p):
            self.get_prob_k_all(l_links, k_links, mu, head, tail, worm_link_coord, dk, df, df_new_head, W, f, num_worms, p_all)
            """Combine both i_worm loops"""
            for i_worm in range(num_worms):
                ###"""p_queue reference or assignment?"""
                ###get_prob_k_one(self, l_links, k_links, mu, head, tail, worm_link_coord, i_worm, dk, df, W, f, p_all[i_worm]):
                if p_draw[i_worm] < p_all[i_worm]:
                    ###accept the move, adapt k and f
                    print("accepted")
                    print(f"worm {i_worm} adjusts link {worm_link_coord[i_worm]}")
                    """INDEX 0 OUT OF BOUNDS???"""
                    print(f"k before {k_links[tuple(worm_link_coord[i_worm])]}")
                    k_links[tuple(worm_link_coord[i_worm])] += dk[i_worm]
                    print(f"k after {k_links[tuple(worm_link_coord[i_worm])]}")
                    print(f"f before {f[tuple(head[i_worm,0])]}")
                    f[tuple(head[i_worm,0])] += df[i_worm]
                    print(f"f after {f[tuple(head[i_worm,0])]}")
                    #k_change[worm_link_coord[i_worm]] += dk[i_worm]
                    #f_change[head[i_worm,0]] += df[i_worm]
                    
                    ##move the head
                    print(f"head before {head[i_worm,0]}")
                    head[i_worm,0] = head[i_worm,1].copy()
                    print(f"head after {head[i_worm,0]}")
                    
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
                #moves[i_worm] = np.array([0,0], dtype=int)
                #dk[i_worm] = 0
                #df[i_worm] = 0
                
    """redundant?"""
    #@jit(nopython=True)
    def update_f_df(self, f, lat_coord, df):
        """
        update f function array with df value
        """
        print(f"updating f[{lat_coord}] += {int(df)}")
        f[tuple(lat_coord)] += f[tuple(lat_coord)] + int(df)
        
    """redundant?"""
    #@jit(nopython=True)
    def update_f_f_change(self, f, f_change):
        """
        update f function array with f_change
        """
        
        f += f_change

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
    
    
    def count_collisions(self, head, hoc_dist_ij, lc_dist_ij, num_worms, np_collisions, collision_worms, non_collision_worms, lc_hoc_bool):
        """
        Count colliding worms in the simulation and separate them into collision "classes"
        can not be paralellized as the counting/separation procedure depends on the previous record
        """
        collisions = []
        
        ##keep track of which worm belongs to which collision
        ##per default they are part of no collision (index=num_worms)
        worm_col_i = np.full(fill_value=num_worms, shape=num_worms,dtype=int)
        
        lc_hoc_bool[0] = False
        lc_hoc_bool[1] = False
        
        ###skippable_worms = np.full(num_worms, fill_value=num_worms, dtype=int)
        
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
                
        ###collision_worms = np.flatten(collisions, dtype=int)
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
                
        np_collisions = np.array(collisions, dtype=int)

    
    """Worm functions"""
    

    #@jit(nopython=True)
    def propose_moves(self, move_dim, move_sign, move_i, move_set, moves, head, worm_link_coord, k_delta, k_links, dk, df, k_change, f_change, num_worms, lat_size, dim):
        """
        Draws a random move from the moveset
        as well as a random value for changing the link value
        and prepares some values needed lateron for the probabalistic sampling step
        """
        
        ##reset k and f change when proposing new moves for all worms
        k_change.fill(0)
        f_change.fill(0)
        
        ##draw a ranom move dimension (dir_i) and orientation (sign)
        #dir_i = np.random.randint(low=0,high=dim)
        move_dim = np.random.randint(low=0, high=dim, size=num_worms)
        print(f"proposing move_dim:\n {move_dim}")
        #sign = np.random.randint(low=0, high=2)
        move_sign = np.random.randint(low=0, high=2, size=num_worms)
        print(f"proposing sign:\n {move_sign}")
        
        ##calculate the corresponding move_index for the move_set
        move_i = move_dim + (dim * move_sign)
        print(f"proposing move_i:\n {move_i}")
        
        ##and move
        for i_worm in range(self.num_worms):
            moves[i_worm] = move_set[ move_i[i_worm] ]
        
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
        head[:,1] += head[:,0] + moves
        
        #print(f"proposing new heads:\n {head[:,1]}")
        
        ##check whether the new head has to be changed for periodic bc
        self.per_lat_coord(head[:,1], lat_size, dim)
        print(f"proposing new heads:\n {head[:,1]}")
        
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
        
        print(f"dk: {dk}")
        print(f"df: {df}")

        ##calculate the k_change array which is then globally added to the lattice 
        ##when doing regular independent sampling of all worms (no collisions!)
        #for i_worm in range(self.num_worms):
        #    k_change[worm_link_coord[i_worm]] += dk[i_worm]
        #    f_change[head[i_worm,0]] += df[i_worm]
    
    """redundant?"""
    #@jit(nopython=True)
    #def extend(head, dir_i, sign, move_set, lat_size, dim):
    def extend(self, head, worm_link_coord, move_i, move_set, lat_size, dim):
        """
        extend the worms head via a move
        """

        print(f"move_set {move_set}")
        print(f"extend old head {head}")
        head += move_set[move_i]
        print(f"extend move {move_set[move_i]}")
        print(f"extend new head {head}")

        self.per_lat_coord(head, lat_size, dim)
        #prop_head = head.copy()
        #print(f"per head {head}")
    
    """redundant?"""
    def set_head(self, head, prop_head):
        """
        set the worms head directly
        """
        print(f"head {head}")
        print(f"prop_head {prop_head}")
        head[:] = prop_head[:]

    """redundant?"""
    #@jit(nopython=True)             
    def check_head_tail(self, check_head, check_tail):
        """
        check whether head and tail coincide
        """
        return np.all(check_head == check_tail)

    #@jit(nopython=True)
    def reset_worm(self, tail, head, worm_link_coord, i_worm, k_delta, lat_size, dim):
        """
        reset the worm randomly
        choose the value of k_delta (fixed!)
        """
        for d in range(dim):
            tail[i_worm,d] = np.random.randint(0,lat_size[d])
            head[i_worm,0,d] = tail[i_worm,d]
            worm_link_coord[d+1] = head[i_worm,0,d]
        worm_link_coord[0] = 0
        """k_delta = np.random.randint(0,1)*2 - 1"""
        k_delta[i_worm] = np.random.randint(0,1)*2 - 1
        print(f"Resetting worm {i_worm} to tail: {tail[i_worm]}, head: {head[i_worm,0]}\n")
        print(f"with k_delta: {k_delta[i_worm]}\n")
        
        
        
        
    """Monte Carlo functions"""

    
    def sweep(self, lat_size, dim, links, sites):
        """
        goal: do a parallel sweep through all given lattice sites (sites)
        """
        
        n_sites = int(np.prod(lat_size))
        

        #save the proposed mcmc modifications in an array
        #rand_mods = np.zeros(conf_size)

        """
        implement checkerboard
        black or white sweep?
        """
        for site in sites:
            """call parallel function mc_site"""


    def draw_site(self, mod_dim, mod_value, dim) :
        """
        choose random modification at random link dimension
        """
        """find better method"""
        mod_dim = np.random.randint(dim)
        mod_value = int(2*np.random.randint(0,1) - 1)

    def mc_site(self, lat_coord, W, f):
        
        """
        MCMC for one single lattice site
        """

        mod_dim = 0
        mod_value = 0
        df = 0
        df_target = 0
        
        ##draw link and link_modification at random
        draw_site(mod_dim, mod_value, dim) 
        
        target_lat_coord = lat_coord + move_set[mod_dim]
        
        """find better method"""
        link_coord = np.concatenate([mod_dim], lat_coord)
        
        ##get acceptance probability
        p = get_prob_l(l_links, k_links, lat_coord, link_coord, target_lat_coord, mod_value, W, f)

        if np.random.uniform < p:
            #accept
            update_link(l_links, link_coord, mod_value)


    #modify to get probability for l change
   #def get_prob_k(l_links, k_links, mu, head, prop_head, tail, sign, move_set, worm_link_coord, k_change, lat_size, dim, W, f):
    def get_prob_l(self, l_links, k_links, mu, link_coord, target_lat_coord, mod_value, W, f):
        p = 1.0

        #change in f function at lat_coord
        """save this change globally?"""
        df = 2 * mod_value

        #change in f function at link_target_coord
        """save this change globally?"""
        df_target = -df

        f_old = f[tuple(link_coord[1:])]
        f_target = f[tuple(target_lat_coord)]

        l_old_link = l_links[tuple(link_coord[1:])]
        l_proposed_link = l_links[tuple(link_coord[1:])] + mod_value
        k_old_link = k_links[tuple(link_coord[1:])]
        print(f"k_old {l_old_link}")
        print(f"k_prop {l_proposed_link}")


        #different factor in acceptance probability for changing modulus of l link
        #accounts for factorial coefficient
        if abs(l_proposed_link) > abs(l_old_link):
            p = p/((abs(k_old_link) + l_proposed_link) * l_proposed_link)
        else:
            p = p*(abs(k_old_link) + l_old_link) * l_old_link

        p *= W[int((f_old + df)/2)]/W[int(f_prop_head/2)]

        return p


    """ SEE ABOVE DEFINITION        
    #@jit(nopython=True)
    def update_link(links, link_coord, sign, value):

        links[tuple(link_coord)] += (1-2*sign) * value
        print(f"updating links[{link_coord}] += {(1-2*sign) * value}")

    """

    """NOT NEEDED
    #function that draws according to the Metropolis algorithm given two probabilities
    def metropolis_p(p_x_new, p_x_old):

        if p_x_new > p_x_old:
            return True
        else:
            if np.random.randint() < (p_x_new/p_x_old):
                return True
            else:
                return False"""
    
    
    
