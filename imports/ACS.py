import numpy as np
from AI2020.IO_manager.io_tsp import TSP_Instance_Creator
from AI2020.solvers.local_search import twoOpt
from AI2020.solvers.constructive_algorithms import nn, compute_lenght
import time
import os


class ACS:
  def __init__(self, parameters):
    # initialisation of ACS info
    self.time_limit = parameters["time_limit"]
    self.ic = parameters["instance"]
    self.n_points = parameters["instance"].nPoints
    self.m = parameters["m"]
    self.q_0 = parameters["q0"]["value"]
    self.beta = parameters["beta"]
    self.alpha = parameters["alpha"]
    self.ro = parameters["ro"]
    self.eta = (self.inverse_distance())**self.beta
    self.Lnn = nn(self.ic.dist_matrix, starting_node = np.random.randint(0,self.n_points))[1]
    self.tau0 = 1./(self.n_points * self.Lnn)
    self.tau = self.tau0 * np.ones([self.n_points,self.n_points])
    self.cl_dim = parameters["cl"] 
    self.cl = self.build_candidate_list()
    self.ants = [self.ant(i) for i in range (self.m)]
    self.ants_state = [0 for i in range(self.m)]
    self.global_best_tour = []
    self.global_best_tours = []
    self.global_best_tour_length = np.inf
    self.best_tours_length = [self.Lnn]
    self.plot = parameters["plot"]
    self.random = parameters["random"] 
    self.ramp = parameters["ramp"] 
    self.ramp_min_ant = parameters["ramp_min_ant"] 
    self.iteration = 0
    np.random.seed(parameters["seed"] )
    self.local_step_info_collector = {k:[] for k in range(self.m)}
    self.pher_eta_collection = {}

  def build_candidate_list(self):
    # creates a dictionary with the candidate list city by city
    cl = {}
    for i in range(self.n_points):
      cl[i] = np.argsort(-self.eta[i])[1:self.cl_dim + 1]
    return cl

  def inverse_distance(self):
    # Matrix with the inverse of the distances. Diagonal filled with np.inf
    kpy = self.ic.dist_matrix.copy()
    np.fill_diagonal(kpy, 1) # to convienently perform the elementwise inverse without having errors      
    eta = 1/kpy
    np.fill_diagonal(eta, np.inf)
    return eta

  def global_update(self):
    self.tau = (1 - self.alpha) * self.tau    #evaporation applied to all edges
    for i in range(self.n_points - 1):        #increase applied only to all edges of the global best tour
        start = self.global_best_tour[i]
        end = self.global_best_tour[i+1]
        self.tau[start][end] += self.alpha * 1/self.global_best_tour_length 
        self.tau[end][start] += self.alpha * 1/self.global_best_tour_length 
    self.start = self.global_best_tour[-1]    #increase applied to the back-home edge (from last to first city of the tour) 
    self.end = self.global_best_tour[0]
    self.tau[start][end] += self.alpha * 1/self.global_best_tour_length 
    self.tau[end][start] += self.alpha * 1/self.global_best_tour_length

  def run(self):
    start = time.time()                             # Start the clock
    while (time.time() - start) < self.time_limit:  # Repeat "iterations" until time limit is reached

      if self.ramp == True and self.iteration>1:     # If ramp == True and after the first iteration,
        self.m = np.max([self.m//2,self.ramp_min_ant])  # the number of ants are halved (integer result) at each iteration
                                                        # being self.ramp_min_ant the min possible headcount.

      self.ants = [self.ant(i) for i in range (self.m)]  #list of m instances of class ant
      self.ants_state = [0 for i in range(self.m)]       #Ants status initialization 0 meaning not completed, 1 completed the tour
      if self.random == True:  # in this case the next ant to move is selected randomly,  regardless the number of steps done so far
        while len(np.where(np.array(self.ants_state) == 0)[0]) != 0:  # until there is a ant that has not completed the tour
          idx_ant = np.random.choice(np.where(np.array(self.ants_state) == 0)[0]) # select an ant randomly among ants which have still not completed their tour
          self.ants[idx_ant].move() # move the selected ant for one step
          self.ants[idx_ant].local_update() # local update performed by the ant that move
      else:
        for step in range(self.n_points - 1): #move all ants by one step
          for ant in self.ants: 
            ant.move()   
          # Local Update
          for ant in self.ants:
            ant.local_update() #local update
      
      tours_lengths = [ant.length_tour for ant in self.ants] 
      best_ant_idx = np.argmin(tours_lengths)  #find the shortes tour in the current iteration
      best_ant = self.ants[best_ant_idx]
      optimal_tour, optimal_length = twoOpt(best_ant.visited_cities, best_ant.length_tour, self.ic.dist_matrix) #TwoOpt of best tour
      self.best_tours_length.append(optimal_length)
      if optimal_length < self.global_best_tour_length: # if optimized tour is better than global one (the best so far)
        self.global_best_tour_length = optimal_length # update global best tour and its length
        self.global_best_tour = optimal_tour
        if self.plot == True: #save info for plots
          self.global_best_tours.append({'optimal_tour': optimal_tour, 'iteration': self.iteration, 'gap': np.round(optimal_length/self.ic.best_sol-1,3)})
      self.global_update() 
      self.iteration += 1

  def ant(self, id): #definition of ant
    network = self #pointer to parent istance of ACS class

    class Ant():
      def __init__(self, id):
        self.id = id
        self.network = network
        self.start = np.random.randint(self.network.n_points)
        self.visited_cities = np.array([self.start])
        self.current_city = self.start
        self.length_tour = 0
        self.local_step_info = []
        self.action = "Exploitation"

      def return_cl(self): 
        #retieve candidate list of current city the ant is in 
        return network.cl[self.current_city]

      def return_cl_cities_left(self, current_city_cl):
        #retieve from candidate list cities not non visitied yet
        return np.setdiff1d(current_city_cl, self.visited_cities) 
      def return_cities_left(self):
        #retieve cities not non visitied yet (not necessarly in the candidate list)
        cities = range(self.network.n_points) 
        return np.setdiff1d(cities, self.visited_cities)

      def exploration(self):
        #return next city the ant should move to, according to exploration logic 
        most_prob_city = np.argsort(-self.network.tau[self.current_city] *
                                    (self.network.eta[self.current_city]))[1]  #city with the higesth tau*eta from current city
        current_city_cl = self.return_cl() 
        cities_left = self.return_cl_cities_left(current_city_cl)
        if len(cities_left) != 1:  #if there are more than 1 city in candidate list not yet visited
          cities_left = np.setdiff1d(cities_left, most_prob_city) #remove the most probable one (it would otherwise be exploitation)
        if len(cities_left) == 0: #if there is not any cities in CL not yet visited
          cities_left = self.return_cities_left() #take the rest of cites (not in candidate list)
          if len(cities_left) != 1:
            cities_left = np.setdiff1d(cities_left, most_prob_city) #remove the most probable one (it would otherwise be exploitation)
        prob = (self.network.tau[self.current_city][cities_left] * 
               (self.network.eta[self.current_city][cities_left])) #array of probabilities depending on tau*eta
        prob_norm = prob/sum(prob) #normalisation 
        return np.random.choice(cities_left, p = prob_norm) #return a city picked with a prob equal to the normalised probability found above

      def exploitation(self):
         #return next city the ant should move to, according to exploitation logic 
        cities_left = self.return_cities_left()
        prob = (self.network.tau[self.current_city][cities_left] * 
               (self.network.eta[self.current_city][cities_left])) #among the non visited city, pick the one with highest tau*eta from current city
        return cities_left[np.argmax(prob)]

      def add_city(self, next_city):
        self.visited_cities = np.append(self.visited_cities, next_city) # add next_city to the list of visited cities

      def next_city(self):
        q = np.random.uniform()             # random float between 0 and 1
        if q <= self.network.q_0:           # if lower than q0=> exploitation
          next_city = self.exploitation()
          self.action = 'Exploitation'
        else:                               # otherwise => exploration
          next_city = self.exploration() 
          self.action = 'Exploration'
        if self.network.plot == True:       # if current simulaiton is to make video, save necesary info (deteriorating perfomances)
          self.update_viz_info(next_city)
        return next_city

      def local_update(self):               # from visited_cities retrive the current and previous cities
        current_city = self.visited_cities[-1]
        prev_city = self.visited_cities[-2]  #update the edge in a symmetric fashion
        self.network.tau[current_city][prev_city] = (1 - self.network.ro) * self.network.tau[current_city][prev_city]
        self.network.tau[current_city][prev_city] += self.network.ro * self.network.tau0
        self.network.tau[prev_city][current_city] = (1 - self.network.ro) * self.network.tau[prev_city][current_city]
        self.network.tau[prev_city][current_city] += self.network.ro * self.network.tau0
        if len(self.visited_cities) == self.network.n_points: #if tour is complete, update also the back home edge (symmetrically) 
          current_city = self.visited_cities[0]
          prev_city = self.visited_cities[-1]
          self.network.tau[current_city][prev_city] = (1 - self.network.ro) * self.network.tau[current_city][prev_city]
          self.network.tau[current_city][prev_city] += self.network.ro * self.network.tau0
          self.network.tau[prev_city][current_city] = (1 - self.network.ro) * self.network.tau[prev_city][current_city]
          self.network.tau[prev_city][current_city] += self.network.ro * self.network.tau0


      def move(self):
        next_city = self.next_city() #pick next city 
        self.length_tour += self.network.ic.dist_matrix[self.current_city][next_city] #update the tour length 
        self.current_city = next_city  #move the ant
        self.add_city(next_city) #add the city to the tour
        if len(self.visited_cities) == self.network.n_points: # if tour is complete
          self.length_tour += self.network.ic.dist_matrix[self.current_city][self.visited_cities[0]] # include the back home edge to the tour length
          self.network.ants_state[self.id] = 1 # update the ant status to 1, mening complete
          if self.network.plot== True: # if the current simulation is for making the video, store visual info for later processing
            self.action = 'End tour, back home'
            self.update_viz_info(self.visited_cities[0], end_tour = True)
            self.network.local_step_info_collector[self.id].append(self.local_step_info)

      def update_viz_info(self, next_city, end_tour = False): 
        # function to save info for plotting 
        fn = str(self.network.iteration)+'_'+str(len(self.local_step_info))
        if not (fn  in self.network.pher_eta_collection): 
          self.network.pher_eta_collection[fn] = (self.network.eta**(self.network.beta)) * self.network.tau

        inf = local_step_info_for_viz(self.current_city,
                                      next_city,
                                      self.visited_cities.copy(),
                                      self.action,
                                      self.network.iteration,
                                      end_tour)
        self.local_step_info.append(inf)

    return Ant(id)

class local_step_info_for_viz:
    #class containing info for plotting
    def __init__(self, current_city, next_city, tour, title, iteration, end_tour):
        self.title = title
        self.next_city = next_city
        self.current_city = current_city
        self.tour = tour
        self.iteration = iteration
        self.end_tour = end_tour


