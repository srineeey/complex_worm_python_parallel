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
        
        self.df_prev_iter = np.zeros((self.num_worms), dtype = int)

        ##save the initial tweak of f for each worm at the start
        ##has to be added later
        self.df0 = np.zeros((self.num_worms), dtype = int)
        
        ##the current changes saved as a lattice "mask"
        self.dk_lat = np.zeros(self.link_size, dtype = int)
        self.df_lat = np.zeros(self.lat_size, dtype = int)
        
        ##traffic light bits that capture which lattice sights are worked on at the moment
        ##new and old heads
        self.old_head_lat = np.zeros(shape=self.lat_size, dtype=bool)
        #self.new_head_lat = np.zeros(shape=self.lat_size, dtype=bool)
        self.work_head_lat = np.zeros(shape=self.lat_size, dtype=bool)
        
        self.tail_df_lat = np.zeros(shape=self.lat_size, dtype=int)
        


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
        self.p_acc = np.ones(self.num_worms, dtype = float)


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


        ##bool value which keeps track whether a worm has closed up
        self.worm_closed = np.full(fill_value = 0, shape=self.num_worms, dtype=bool)

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

        print(f"white sites: {self.n_w_sites}")
        print(f"black sites: {self.n_b_sites}")

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
        
        self.mcmc_w_p_acc = np.ones(self.n_w_sites)
        self.mcmc_b_p_acc = np.ones(self.n_b_sites)

    
    def run_i_worm(self,
                   i_worm,
                   k_links, l_links, mu, W, f,
                   old_head_lat, work_head_lat, tail_df_lat,
                   move_dim, move_sign, move_i, move_set, moves,
                   head, tail, worm_link_coord, worm_closed,
                   k_delta, dk, df, df_prev_iter, df0,
                   p_acc,
                   num_worms, lat_size, dim):
        
        while not worm_closed[i_worm]:
        
            #flip old head bit
            """ONLY ACCESS ON WORM AT A TIME"""
            if work_head_lat[tuple(head[i_worm,0])] == 1:
                ##something has gone terribly wrong
                raise ValueError("work head bit is set to 1, even though it should be 0!")

            #work_head_lat[tuple(head[i_worm,0])] = 1

            self.propose_move_i_worm(i_worm,
                                     k_links,
                                     move_dim, move_sign, move_i, move_set, moves,
                                     head, worm_link_coord,
                                     k_delta, dk, df, df_prev_iter,
                                     lat_size, dim)

            #if work_head_lat[tuple(head[i_worm,1])] == 1 :
            if work_head_lat[tuple(head[i_worm,1])] == 1 and work_head_lat[tuple(head[i_worm,0])] == 1:
                print("worm already working!")
                ##wait
                ##stop worm in this iteration
            else:
                ##flip new head bit
                """ONLY ACCESS ON WORM AT A TIME"""
                work_head_lat[tuple(head[i_worm,1])] = 1
                work_head_lat[tuple(head[i_worm,0])] = 1

                ##continue sampling
                ##+reset new and old head lat bit
                self.sample_k_i_worm(i_worm,
                                    k_links, l_links, mu, W, f,
                                    old_head_lat, work_head_lat, tail_df_lat,
                                    moves,
                                    head, tail, worm_link_coord, worm_closed,
                                    dk, df, df_prev_iter,
                                    p_acc,
                                    lat_size, dim
                                    )

            
        print(f"worm {i_worm} has closed up")
        #self.reset_worm_i_worm(i_worm,
        #                      k_links, l_links, mu, W, f,
        #                      old_head_lat, work_head_lat, tail_df_lat,
        #                      move_dim, move_sign, move_i, move_set, moves,
        #                      head, tail, worm_link_coord, worm_closed,
        #                      k_delta, dk, df, df0,
        #                      p_acc,
        #                      num_worms, lat_size, dim
        #                      )
            
            
    #@jit(nopython=True)
    def reset_worm_i_worm(self,
                          i_worm,
                          k_links, l_links, mu, W, f,
                          old_head_lat, work_head_lat, tail_df_lat,
                          move_dim, move_sign, move_i, move_set, moves,
                          head, tail, worm_link_coord, worm_closed,
                          k_delta, dk, df, df_prev_iter, df0,
                          p_acc,
                          num_worms, lat_size, dim):
        

        """
        write initial df0
        reset the worm randomly
        choose the value of k_delta (fixed!)
        """
        
        print("resetting worm")
        
        tail_df_lat[tuple(tail[i_worm])] = 0
        worm_closed[i_worm] = 1
        print("f before:")
        print(f)
        f[tail[i_worm]] += df0[i_worm]
        print("f after:")
        print(f)
        k_delta[i_worm] = np.random.randint(low=0, high=2)*2 - 1
        
        df_prev_iter[i_worm] = 0
            
        ##reset the worm randomly
        ##find a new head and tail
        ##that does not coincide with other worms
        for d in range(dim):
            tail[i_worm,d] = np.random.randint(low=0, high=lat_size[d])
            head[i_worm,0,d] = tail[i_worm,d]


        """ONLY ACCESS ON WORM AT A TIME"""
        while old_head_lat[tuple(head[i_worm,0])]:
            for d in range(dim):
                tail[i_worm,d] = np.random.randint(low=0, high=lat_size[d])
                head[i_worm,0,d] = tail[i_worm,d]

        ## it can start working at this head!
        #flip old head bit            
        old_head_lat[tuple(head[i_worm,0])] = 1
        
        ##continue sampling until the worm starts and opens up
        while worm_closed[i_worm]:
            
            #propose an initial move
            self.propose_move_i_worm(i_worm,
                                     k_links,
                                     move_dim, move_sign, move_i, move_set, moves,
                                     head, worm_link_coord,
                                     k_delta, dk, df, df_prev_iter,
                                     lat_size, dim)

            #check whether this worm can modify the link safely
            if work_head_lat[tuple(head[i_worm,1])] == 1:
            #if work_head_lat[tuple(head[i_worm,1])] == 1 and work_head_lat[tuple(head[i_worm,0])] == 1:
                print("worm already working!")
                ##no it can not!
                ##other worm is already working there
            else:
                ##yes it can!
                ##flip new head bit
                """ONLY ACCESS ON WORM AT A TIME"""
                work_head_lat[tuple(head[i_worm,1])] = 1

                ##continue sampling
                ##+reset new and old head lat bit
                self.sample_k_i_worm_start(i_worm,
                                            k_links, l_links, mu, W, f,
                                            old_head_lat, work_head_lat, tail_df_lat,
                                            moves,
                                            head, tail, worm_link_coord, worm_closed,
                                            dk, df, df0, df_prev_iter,
                                            p_acc,
                                            lat_size, dim
                                          )

                         
                         
        print(f"Resetting worm {i_worm} to tail: {tail[i_worm]}, head: {head[i_worm,0]}\n")
        print(f"with k_delta: {k_delta[i_worm]}\n")



        
        

            
        #@jit(nopython=True)
    def propose_move_i_worm(self,
                            i_worm,
                            k_links,
                            move_dim, move_sign, move_i, move_set, moves,
                            head, worm_link_coord,
                            k_delta, dk, df, df_prev_iter,
                            lat_size, dim):
        """
        Draws a random move from the moveset
        as well as a random value for changing the link value
        and prepares some values needed lateron for the probabalistic sampling step
        """


        ##draw a ranom move dimension (dir_i) and orientation (sign)
        move_dim[i_worm] = np.random.randint(low=0, high=dim)
        print(f"proposing move_dim:\n {move_dim}")
        move_sign[i_worm] = np.random.randint(low=0, high=2)
        print(f"proposing sign:\n {move_sign}")

        ##calculate the corresponding move_index for the move_set
        move_i[i_worm] = move_dim[i_worm] + (dim * move_sign[i_worm])
        print(f"proposing move_i:\n {move_i}")

        ##and move
        moves[i_worm] = move_set[ move_i[i_worm] ]

        #print(f"proposing moves:\n {moves}")

        ##the link value which is to be modified if the move is accepted
        #worm_link_coord[0] = dir_i
        worm_link_coord[i_worm,0] = move_dim[i_worm]
        worm_link_coord[i_worm,1:] = head[i_worm,0].copy()
        ##check whether the link index has to be changed in order to be accessed
        self.transform_link_index(worm_link_coord[i_worm], move_sign[i_worm], moves[i_worm], lat_size, dim)

        ##the new proposed head if the move is accepted
        head[i_worm,1] = head[i_worm,0] + moves[i_worm]

        print(f"proposing new heads:\n {head[:,1]}")

        ##check whether the new head has to be changed for periodic bc
        self.per_lat_coord(head[i_worm,1], lat_size, dim)

        print(f"proposing new heads:\n {head[:,1]}")

        old_k = k_links[tuple(worm_link_coord[i_worm])]

        """NOT CORRECTLY SETTING DF"""
        ##changes which should be kept track of for each worm
        dk_p = (1-2*move_sign[i_worm]) * k_delta[i_worm]
        #if np.sign(dk_p) == np.sign(old_k):
        df_p = (abs(old_k+dk_p) - abs(old_k))

        dk[i_worm] = dk_p
        df[i_worm] = df_p
        #df_prev_iter[i_worm] = df_p
        
        print(f"k_old: {old_k}")
        print(f"dk: {dk}")
        print(f"df: {df}")

        ##calculate the k_change array which is then globally added to the lattice
        ##when doing regular independent sampling of all worms (no collisions!)
        #for i_worm in range(self.num_worms):
        #    k_change[worm_link_coord[i_worm]] += dk[i_worm]
        #    f_change[head[i_worm,0]] += df[i_worm]
        

        
           
    """Parallelize completely"""
    def sample_k_i_worm(self,
                        i_worm,
                        k_links, l_links, mu, W, f,
                        old_head_lat, work_head_lat, tail_df_lat,
                        moves,
                        head, tail, worm_link_coord, worm_closed,
                        dk, df, df_prev_iter,
                        p_acc,
                        lat_size, dim):

            p_acc[i_worm] = 1.
            p_draw = np.random.uniform(low=0., high= 1.)

            df_new_head = tail_df_lat[tuple(head[i_worm,1])]

            #print("Starting Metropolis algorithm for all worms")
            #print(f"dk: {dk}")
            #print(f"df: {df}")
            #print(f"df_new_head: {df_new_head}")
            #( l_links, k_links, mu, head, tail, worm_link_coord, dk, df, df_new_head, W, f, num_worms, p):
            """CHECK HEAD MEETS TAIL"""
            self.get_prob_k_i_worm(i_worm,
                                   k_links, l_links, mu, W, f,
                                   head, worm_link_coord,
                                   dk, df, df_new_head, df_prev_iter,
                                   p_acc)
            
            print(f"p_acc: {p_acc[i_worm]}")
            print(f"p_draw: {p_draw}")
            bool_acc = int( (1 + np.sign(p_acc[i_worm] - p_draw))*0.5 )
            print(f"accepted: {bool_acc}")

            ###accept the move, adapt k and f

            #print(f"accepted: {bool_acc}")
            #print(f"worm {i_worm} adjusts link {worm_link_coord[i_worm]}")
            #print(f"k before {k_links[tuple(worm_link_coord[i_worm])]}")
            k_links[tuple(worm_link_coord[i_worm])] += bool_acc*dk[i_worm]
            #print(f"k after {k_links[tuple(worm_link_coord[i_worm])]}")
            #print(f"f before {f[tuple(head[i_worm,0])]}")
            
            """PROBLEM OF SAVING DF EVEN THOUGH MOVE IS NOT ACCEPTED"""
            #change f at old head
            print("f before:")
            print(f)
            #f[tuple(head[i_worm,0])] += (bool_acc * df[i_worm]) + df_prev_iter[i_worm]
            f[tuple(head[i_worm,0])] += bool_acc * (df[i_worm] + df_prev_iter[i_worm])
            print("f after:")
            print(f)
            df_prev_iter[i_worm] = (1-bool_acc) * df_prev_iter[i_worm]
            
            df_prev_iter[i_worm] += (bool_acc * df[i_worm])
            print(f"saving df_prev_iter[i_worm] {df_prev_iter[i_worm]}")
            #print(f"f after {f[tuple(head[i_worm,0])]}")
            
            ##once changes have been made
            ##reset old head lat bit safely
            #work_head_lat[tuple(head[i_worm,1])] = bool(bool_acc)
            #work_head_lat[tuple(head[i_worm,0])] = bool(1 - bool_acc)
            
            work_head_lat[tuple(head[i_worm,1])] = 0
            work_head_lat[tuple(head[i_worm,0])] = 0
            
            print(f"Resetting bits at prop_head {head[i_worm,1]}={work_head_lat[tuple(head[i_worm,1])]} and old_head {head[i_worm,0]}={work_head_lat[tuple(head[i_worm,0])]}")

            ##move the head
            #print(f"head before {head[i_worm,0]}")
            head[i_worm,0] += bool_acc*moves[i_worm]
            self.per_lat_coord(head[i_worm,0], lat_size, dim)
            #print(f"head after {head[i_worm,0]}")
            
            #work_head_lat[tuple(head[i_worm,0])] = 0
            #work_head_lat[tuple(head[i_worm,0])] = 1

            ##check whether head == tail
            ##and save it
            #worm_closed[i_worm] = np.all( head[i_worm,0] == tail[i_worm] )
            print(f"head: {head[i_worm,0]}, tail: {tail[i_worm]}")
            
            print(f"worm closed: {worm_closed[i_worm]}")
            worm_closed[i_worm] = bool(1 - np.sign( ((head[i_worm,0] - tail[i_worm])**2).sum() ))
            print(f"worm closed: {worm_closed[i_worm]}")
            ##reset prop head to old head
            head[i_worm,1] = head[i_worm,0].copy()
            #moves[i_worm] = np.array([0,0], dtype=np.int64)
            
            
            
    def get_prob_k_i_worm(self,
                           i_worm,
                           k_links, l_links, mu, W, f,
                           head, worm_link_coord,
                           dk, df, df_new_head, df_prev_iter,
                           p_acc):
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
        p_acc[i_worm] = 1.

        #print(f"Calculating acceptance probability for worm {i_worm}")


        ##just some values needed for acceptance probability
        #for i_worm in range(num_worms):
        k_old_link = k_links[tuple(worm_link_coord[i_worm])]
        k_proposed_link = k_links[tuple(worm_link_coord[i_worm])] + dk[i_worm]
        l_old_link = l_links[tuple(worm_link_coord[i_worm])]

        f_old = f[tuple(head[i_worm,0])]
        f_prop_head = f[tuple(head[i_worm,1])]
                
        print(f"f_old {f[tuple(head[i_worm,0])]}")
        print(f"df[i_worm] {df[i_worm]}")
        print(f"f_prop_head {f[tuple(head[i_worm,1])]}")
        print(f"df_prop_head {df_new_head}")
        

        ##different factor in acceptance probability for changing modulus of k link
        ##accounts for factorial coefficient in formula
        """REPLACE IF STATEMENT BY EXPRESSION"""
        
        print(f"l_old_link {l_old_link}")
        print(f"k_proposed_link {k_proposed_link}")
        print(f"k_old_link {k_old_link}")
        print(f"(abs(k_proposed_link) + l_old_link) {(abs(k_proposed_link) + l_old_link)}")
        
        print(f"p_acc[i_worm] {p_acc[i_worm]}")
        
        if abs(k_proposed_link) > abs(k_old_link):
            p_acc[i_worm] = p_acc[i_worm] / float((abs(k_proposed_link) + l_old_link))
        else:
            p_acc[i_worm] = p_acc[i_worm] * float((abs(k_old_link) + l_old_link))

        print(f"p_acc[i_worm] {p_acc[i_worm]}")
        print(f"int((f_old + df[i_worm] + df_prev_iter[i_worm])) {int((f_old + df[i_worm] + df_prev_iter[i_worm]))}")
        print(f"int((f_prop_head + df_new_head) {int((f_prop_head + df_new_head))}")

        print(f"W[int((f_old + df[i_worm] + df_prev_iter[i_worm])//2)] {W[int((f_old + df[i_worm] + df_prev_iter[i_worm])//2)]}")
        print(f"W[int((f_prop_head + df_new_head)//2)] {W[int((f_prop_head + df_new_head)//2)]}")
        ##multiply p with W[f_prop/2]
        #p_acc[i_worm] *= W[int((f_old + df)//2)]/W[int((f_prop_head + df_new_head)//2)]
        p_acc[i_worm] *= W[int((f_old + df[i_worm] + df_prev_iter[i_worm])//2)]/W[int((f_prop_head + df_new_head)//2)]
        
        print(f"p_acc[i_worm] {p_acc[i_worm]}")


        ###p *= ( (1 - int(np.all(old_head == tail)))*W[int((f_old + df)//2)] + int(np.all(old_head == tail)) )/W[int(f_prop_head//2)]

        ##if direction is timelike (dir_i = 0) multiply by exp(-change*mu)
        #if worm_link_coord[i_worm][0] == 0:
        #    #p *= np.exp((1. - 2.*sign)*mu*value)
        #    p *= np.exp(float(dk)*mu)
        print(f"float(dk) {float(dk)}")
        print(f"mu {mu}")
        print(f"np.exp( (1-np.sign(worm_link_coord[i_worm][0])) * float(dk) * mu ) {np.exp( (1-np.sign(worm_link_coord[i_worm][0])) * float(dk) * mu )}")
        #p_acc[i_worm] *= np.exp( np.sign(worm_link_coord[i_worm][0])*float(dk)*mu )
        p_acc[i_worm] *= np.exp( (1-np.sign(worm_link_coord[i_worm][0])) * float(dk) * mu )
        
        print(f"p_acc[i_worm] {p_acc[i_worm]}")

        #print(f"k_old {k_old_link}")
        #print(f"dk {dk}")
        #print(f"k_prop {k_proposed_link}")
        #print(f"df {df}")
        #print(f"df_new_head {df_new_head}")
        #print(f"p {p}")
                         
                         
                         
                        
    def sample_k_i_worm_start(self,
                            i_worm,
                            k_links, l_links, mu, W, f,
                            old_head_lat, work_head_lat, tail_df_lat,
                            moves,
                            head, tail, worm_link_coord, worm_closed,
                            dk, df, df_prev_iter, df0,
                            p_acc,
                            lat_size, dim):

            p_acc[i_worm] = 1
            p_draw = np.random.uniform(low=0., high= 1.)

            df_new_head = tail_df_lat[tuple(head[i_worm,1])]

            #print("Starting Metropolis algorithm for all worms")
            #print(f"dk: {dk}")
            #print(f"df: {df}")
            #print(f"df_new_head: {df_new_head}")
            #( l_links, k_links, mu, head, tail, worm_link_coord, dk, df, df_new_head, W, f, num_worms, p):

            self.get_prob_k_i_worm_start(i_worm,
                                         k_links, l_links, mu, W, f,
                                         head, worm_link_coord,
                                         dk, df, df_new_head, df_prev_iter,
                                         p_acc)
            
            bool_acc = int( (1 + np.sign(p_acc[i_worm] - p_draw))*0.5 )

            ###accept the move, adapt k and f

            #print(f"accepted: {bool_acc}")
            #print(f"worm {i_worm} adjusts link {worm_link_coord[i_worm]}")
            #print(f"k before {k_links[tuple(worm_link_coord[i_worm])]}")
            k_links[tuple(worm_link_coord[i_worm])] += bool_acc*dk[i_worm]
            #print(f"k after {k_links[tuple(worm_link_coord[i_worm])]}")
            #print(f"f before {f[tuple(head[i_worm,0])]}")
            """CHANGE F AT HEAD AND NEW HEAD"""
            ##rememebr the initial df change at tail
            df0[i_worm] += bool_acc*df[i_worm]
            
            
            df_prev_iter[i_worm] = (1-bool_acc) * df_prev_iter[i_worm]
            
            df_prev_iter[i_worm] += (bool_acc * df[i_worm])
            
            
            #print(f"f after {f[tuple(head[i_worm,0])]}")
            print(f"saving df_prev_iter[i_worm] {df_prev_iter[i_worm]}")
            ##reset new head lat bit
            ##once changes have been made
            ##reset old head lat bit safely
            #work_head_lat[tuple(head[i_worm,1])] = bool(bool_acc)
            #work_head_lat[tuple(head[i_worm,0])] = bool(1 - bool_acc)
            
            work_head_lat[tuple(head[i_worm,1])] = 0
            work_head_lat[tuple(head[i_worm,0])] = 0
            
            print(f"Resetting bits at prop_head {head[i_worm,1]}={work_head_lat[tuple(head[i_worm,1])]} and old_head {head[i_worm,0]}={work_head_lat[tuple(head[i_worm,0])]}")

            tail_df_lat[tuple(head[i_worm,0])] += bool_acc*df0[i_worm]
            
            ##move the head
            #print(f"head before {head[i_worm,0]}")
            head[i_worm,0] += bool_acc*moves[i_worm]
            self.per_lat_coord(head[i_worm,0], lat_size, dim)
            #print(f"head after {head[i_worm,0]}")
            
            #work_head_lat[tuple(head[i_worm,0])] = 0
            #work_head_lat[tuple(head[i_worm,0])] = 1

            ##check whether worm has moved
            ##and save it
            #worm_closed[i_worm] = np.all( head[i_worm,0] == tail[i_worm] )
            #worm_closed[i_worm] = np.sign( ((head[i_worm,0] - tail[i_worm])**2).sum() )
            print(f"worm closed: {worm_closed[i_worm]}")
            worm_closed[i_worm] = 1 - bool_acc
            print(f"worm closed: {worm_closed[i_worm]}")
                         
            ##reset prop head to old head
            head[i_worm,1] = head[i_worm,0].copy()
            #moves[i_worm] = np.array([0,0], dtype=np.int64)
            
            
            
            
            
                  
    def get_prob_k_i_worm_start(self,
                                i_worm,
                                k_links, l_links, mu, W, f,
                                head, worm_link_coord,
                                dk, df, df_new_head, df_prev_iter,
                                p_acc):
        

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


        ##just some values needed for acceptance probability
        #for i_worm in range(num_worms):
        k_old_link = k_links[tuple(worm_link_coord[i_worm])]
        k_proposed_link = k_links[tuple(worm_link_coord[i_worm])] + dk[i_worm]
        l_old_link = l_links[tuple(worm_link_coord[i_worm])]

        f_old = f[tuple(head[i_worm,0])]
        f_prop_head = f[tuple(head[i_worm,1])]
        
        print(f"f_old {f[tuple(head[i_worm,0])]}")
        print(f"df[i_worm] {df[i_worm]}")
        print(f"f_prop_head {f[tuple(head[i_worm,1])]}")

        ##different factor in acceptance probability for changing modulus of k link
        ##accounts for factorial coefficient in formula
        
        print(f"l_old_link {l_old_link}")
        print(f"k_proposed_link {k_proposed_link}")
        print(f"k_old_link {k_old_link}")
        print(f"(abs(k_proposed_link) + l_old_link) {(abs(k_proposed_link) + l_old_link)}")
        
        print(f"p_acc[i_worm] {p_acc[i_worm]}")
        
        if abs(k_proposed_link) > abs(k_old_link):
            p_acc[i_worm] = p_acc[i_worm] / (abs(k_proposed_link) + l_old_link)
        else:
            p_acc[i_worm] = p_acc[i_worm] * (abs(k_old_link) + l_old_link)

        print(f"p_acc[i_worm] {p_acc[i_worm]}")
        print(f"int((f_prop_head + df_new_head))] {int((f_prop_head + df_new_head))}")
        
        print(f"W[int((f_prop_head + df_new_head)//2)] {W[int((f_prop_head + df_new_head)//2)]}")

        ##multiply p with W[f_prop/2]
        p_acc[i_worm] *= 1./W[int((f_prop_head + df_new_head)//2)]

        print(f"p_acc[i_worm] {p_acc[i_worm]}")
        
        ###p *= ( (1 - int(np.all(old_head == tail)))*W[int((f_old + df)//2)] + int(np.all(old_head == tail)) )/W[int(f_prop_head//2)]

        ##if direction is timelike (dir_i = 0) multiply by exp(-change*mu)
        #if worm_link_coord[i_worm][0] == 0:
        #    #p *= np.exp((1. - 2.*sign)*mu*value)
        #    p *= np.exp(float(dk)*mu)
        
        print(f"float(dk) {float(dk)}")
        print(f"mu {mu}")
        print(f"np.exp( (1-np.sign(worm_link_coord[i_worm][0])) * float(dk) * mu ) {np.exp( (1-np.sign(worm_link_coord[i_worm][0])) * float(dk) * mu )}")
            
        p_acc[i_worm] *= np.exp( (1-np.sign(worm_link_coord[i_worm][0])) * float(dk) * mu )
        #p_acc[i_worm] = p_acc[i_worm]* (np.exp( (1-np.sign(worm_link_coord[i_worm][0])) * float(dk) * mu ))
        
        print(f"p_acc[i_worm] {p_acc[i_worm]}")

        #print(f"k_old {k_old_link}")
        #print(f"dk {dk}")
        #print(f"k_prop {k_proposed_link}")
        #print(f"df {df}")
        #print(f"df_new_head {df_new_head}")
        #print(f"p {p}")
        



        

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
    def transform_link_index(self, link_coord, move_sign, move, lat_size, dim):
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
        link_coord[1:] += move_sign * move
        #link_coord[1:] = link_coord[1:] + (sign * move_set[move_i])

        ##impose periodic boundary conditions
        #self.per_lat_coord(link_coord[1:], lat_size, dim)
        ##for multiple worms
        self.per_lat_coord(link_coord[1:], lat_size, dim)

        #link_index = [dir_i] + new_lat_coord
        #print(f"valid link index:\n {link_coord}")
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

        #for d in np.arange(start=0, stop=dim, step=1, dtype=np.int64):
        for d in range(0, int(dim), 1):
        #for d in range(0, len(lat_size), 1):
            #if lat_coord is outside of the possible index domain

            #if lat_coord[d] >= lat_size[d]:
            #    lat_coord[d] -= lat_size[d]
            #elif lat_coord[d] < 0:
            #    lat_coord[d] += lat_size[d]
                
            lat_coord[d] = (lat_coord[d] + lat_size[d])%lat_size[d]
            
        #print(f"lat_coord after bc: \n {lat_coord}")

        #return lat_coord

    
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





    """Monte Carlo functions"""

    def sweep(self,
              sites, target_sites, n_sites,
              k_links, l_links, W, f,
              link_coord, mod_dim, dl, df, df_target,
              move_set, moves,
              p_acc,
              lat_size, dim,
              ):
        """
        goal: do a parallel sweep through all given lattice sites
        """

        for i_site in range(n_sites):
            self.mc_site(i_site, sites, target_sites,
                         k_links, l_links, W, f,
                         link_coord, mod_dim, dl, df, df_target,
                         move_set, moves,
                         p_acc,
                         lat_size, dim
                        )
        


    def mc_site(self,
                i_site, sites, target_sites,
                k_links, l_links, W, f,
                link_coord, mod_dim, dl, df, df_target,
                move_set, moves,
                p_acc,
                lat_size, dim
               ):

        """
        MCMC for one single lattice site
        """
        
        print(f"sampling site {sites[i_site]}")

        p_draw = np.random.uniform(low=0., high=1.)
        ##p = np.ones(1)
        p_acc[i_site] = 1.

        ##draw link and link_modification at random
        self.draw_site(i_site, sites, target_sites,
                       l_links,
                       link_coord, mod_dim, dl, df, df_target,
                       move_set, moves,
                       lat_size, dim,
                      )


        ##get acceptance probability
        self.get_prob_l_site(i_site, sites, target_sites,
                            k_links, l_links, W, f,
                            link_coord,
                            dl, df, df_target,
                            p_acc
                            )

        #if p_draw[i_site] < p[i_site]:
        #if p_draw[i_site] < 0.5:
        #if True:
        #    #accept
        #    #print("accepted")
        #    l_links[tuple(link_coord[i_site])] += dl[i_site]
        #    f[tuple(sites[i_site])] += df[i_site]
        #    f[tuple(target_sites[i_site])] += df_target[i_site]
        #else:
            #print("rejected")
        #    pass

        print(f"p_acc: {p_acc[i_site]}")
        print(f"p_draw: {p_draw}")
        bool_acc = int( (1 + np.sign(p_acc[i_site] - p_draw))*0.5 )
        print(f"accpeted: {bool_acc}")
        print("f before:")
        print(f)
        print(f"modifying link {link_coord[i_site]}, value {l_links[tuple(link_coord[i_site])]}")
        print(f"with {dl[i_site]}")
        l_links[tuple(link_coord[i_site])] += bool_acc*dl[i_site]
        f[tuple(sites[i_site])] += bool_acc*df[i_site]
        f[tuple(target_sites[i_site])] += bool_acc*df_target[i_site]
        
        print("f after:")
        print(f)
        
        print(f"after sampling link value at site {link_coord[i_site]}: {l_links[tuple(link_coord[i_site])]}")
        
        

    def draw_site(self,
                  i_site, sites, target_sites,
                  l_links,
                  link_coord, mod_dim, dl, df, df_target,
                  move_set, moves,
                  lat_size, dim,
                 ):
        """
        choose random modification at random link dimension
        """
        """find better method"""
        mod_dim[i_site] = np.random.randint(low=0, high=dim)
        print(f"mod_dim: {mod_dim[i_site]}")
        #if l_links[tuple(link_coord[i_site])] == 0:
        #    dl[i_site] = 1
        #else:
        #    dl[i_site] = (2*np.random.randint(low=0, high=2) - 1)
        
        dl[i_site] = int( 1 - 2*np.sign(l_links[tuple(link_coord[i_site])])*np.random.randint(low=0, high=2) )
        print(f"dl[i_site]: {dl[i_site]}")
        
        df[i_site] = 2*dl[i_site]
        #df_target[i_site] = -df[i_site]
        """l links have no direction?"""
        df_target[i_site] = df[i_site]
        link_coord[i_site,0] = mod_dim[i_site]
        link_coord[i_site,1:] = sites[i_site]
        
        print(f"df[i_site]: {df[i_site]}")
        print(f"df_target[i_site]: {df_target[i_site]}")
        
        moves[i_site] = move_set[mod_dim[i_site]] 
        
        target_sites[i_site] = sites[i_site] + moves[i_site]

        #(lat_coord, lat_size, dim):
        self.per_lat_coord(target_sites[i_site], lat_size, dim)
        print(f"target_sites[i_site]: {target_sites[i_site]}")



    def get_prob_l_site(self, 
                        i_site, sites, target_sites,
                        k_links, l_links, W, f,
                        link_coord,
                        dl, df, df_target,
                        p_acc,
                       ):
        p_acc[i_site] = 1.0

        f_old = f[tuple(sites[i_site])]
        f_target = f[tuple(target_sites[i_site])]

        l_old_link = l_links[tuple(link_coord[i_site])]
        l_proposed_link = l_links[tuple(link_coord[i_site])] + dl[i_site]
        k_old_link = k_links[tuple(link_coord[i_site])]
        
        print(f"f_old at site {sites[i_site]}: {f_old}")
        print(f"f_target at site {target_sites[i_site]}: {f_target}")
        print(f"df at site {sites[i_site]}: {df[i_site]}")
        
        #different factor in acceptance probability for changing modulus of l link
        #accounts for factorial coefficient
        
        print(f"k_old_link {k_old_link}")
        print(f"l_proposed_link {l_proposed_link}")
        print(f"l_old_link {l_old_link}")
        
        print(f"p_acc[i_site] {p_acc[i_site]}")
        
        print(f"(abs(k_old_link) + l_proposed_link) * l_proposed_link) {(abs(k_old_link) + l_proposed_link) * l_proposed_link}")
        print(f"(abs(k_old_link) + l_old_link) * l_old_link {(abs(k_old_link) + l_old_link) * l_old_link}")
        
        """REPLACE IF STATEMENT WITH EXPRESSION"""
        if abs(l_proposed_link) > abs(l_old_link):
            p_acc[i_site] = p_acc[i_site]/((abs(k_old_link) + l_proposed_link) * l_proposed_link)
        else:
            p_acc[i_site] = p_acc[i_site]*(abs(k_old_link) + l_old_link) * l_old_link
        
        print(f"p_acc[i_site] {p_acc[i_site]}")
        
        
        print(f"int((f_old + df[i_site])/2) {int((f_old + df[i_site])/2)}")
        print(f"int((f_target + df_target[i_site])/2) {int((f_target + df_target[i_site])/2)}")
        print(f"int(f_old/2) {int(f_old/2)}")
        print(f"int((f_target)/2) {int((f_target)/2)}")
        print(f"W[int((f_old + df[i_site])/2)] {W[int((f_old + df[i_site])/2)]}")
        print(f"W[int((f_target + df_target[i_site])/2)] {W[int((f_target + df_target[i_site])/2)]}")
        print(f"W[int(f_old/2)] {W[int(f_old/2)]}")
        print(f"W[int((f_target)/2)] {W[int((f_target)/2)]}")
        
        
        """df_target?"""
        p_acc[i_site] *= ( W[int((f_old + df[i_site])/2)] * W[int((f_target + df_target[i_site])/2)] )/( W[int(f_old/2)] * W[int((f_target)/2)] )
        ###p_acc[i_site] *= W[int((f_old + df[i_site])/2)]/W[int(f_target/2)]

        print(f"p_acc[i_site] {p_acc[i_site]}")
        

        
        
        
        
        
    def mc_sites(self,
                 sites, target_sites, link_coord,
                 mod_dim, dl, df, df_target, 
                 l_links, k_links, W, f,
                 n_sites,
                 move_set, moves,
                 lat_size, dim):

        """
        MCMC for multiple lattice sites
        """

        p_draw = np.random.uniform(low=0., high=1., size=n_sites)
        p = np.ones(len(sites))

        ##draw link and link_modification at random
        self.draw_sites(mod_dim, dl, df, df_target, link_coord, sites, target_sites, l_links, move_set, moves, n_sites, lat_size, dim)

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


    def draw_sites(self,
                   sites, target_sites,
                   l_links,
                   link_coord, mod_dim, dl, df, df_target,
                   move_set, moves,
                   lat_size, dim
                  ):
        """
        choose random modifications at random link dimensions
        """
        mod_dim = np.random.randint(low=0, high=dim, size=n_sites)
        
        #dl[:] = (2*np.random.randint(low=0, high=2, size=n_sites) - 1)[:]

        ##l links > 0
        ##dl must be > 0 if l = 0
        for i_site in range(n_sites):
            #if l_links[tuple(link_coord[i_site])] == 0:
            #    dl[i_site] = 1
            #moves[i_site] = move_set[mod_dim[i_site]]
            dl[i_site] = int (1 - 2*np.sign(l_links[tuple(link_coord[i_site])])*np.random.randint(low=0, high=2) )
            
        #print(f"MCMC dl:\n{dl}")
        """CHECK WHETHER CORRECT"""
        df[:] = 2*dl[:]
        df_target[:] = df[:]
        link_coord[:,0] = mod_dim.copy()
        link_coord[:,1:] = sites.copy()

        target_sites[:] = (sites + moves)[:]

        #(lat_coord, lat_size, dim):
        self.per_lat_coord(target_sites, lat_size, dim)

        #print(f"MCMC link_coord:\n{link_coord}")
        #print(f"MCMC target site:\n{target_sites}")