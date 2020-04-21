from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit

from constants import HEIGHT, WIDTH, MAX_RADIUS, N_CONCEPTS, N_FEATURES, NOISE_RATE, LEARNING_RATE, SAMPLE, HSAMPLE, N_AGENTS
import itertools


class Agent(Agent):
    def __init__(self, pos, model):
        '''
         Create a new speech agent.

         Args:
            pos: Agent initial location.
            model: Model in which agent is located.
        '''
        super().__init__(pos, model)
        self.pos = pos
        self.language_agg = [0, 0, 0]

        # Initialize array of concepts
        self.concepts = np.arange(0, N_CONCEPTS)
        # Initialize array of of language: draw binary features from uniform distribution
        self.language = np.random.randint(0,2, (N_CONCEPTS, N_FEATURES))
        self.language_agg = self.compute_language_agg(self.language)  

    def compute_language_agg(self, language):
        # Only look at first three concepts: every concept will be a channel
        # Scale by total possible sum
        color_scale = 255
        return language[1,:3].clip(0).clip(max=1) * color_scale


    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
        # Choose an agent to speak to
        # If density==1 and radius==MAX_RADIUS, every agent speaks with every other, so random mixing
        neighbors = self.model.grid.get_neighbors(self.pos, True, False, self.model.radius)
        listener = np.random.choice(neighbors)
        self.speak(listener)
    
    ### Methods used when agent speaks
    
    def speak(self, listener):
        '''
         Speak to other agent

         Args:
            listener: agent to speak to
        '''
        concept = np.random.choice(self.concepts)

        signal = np.copy(self.language[concept]) # create copy, so noise not applied to original
        if NOISE_RATE > 0.0:
            print("Apply noise")
            # Apply noise by replacing some bits
            replace_positions = np.random.choice([True,False], size=N_FEATURES, p=[NOISE_RATE,1-NOISE_RATE])
            random_vector = np.random.randint(0,2,N_FEATURES)
            signal.put(replace_positions, random_vector)


        # (S4) Send to listener, and receive concept listener points to
        concept_listener = listener.listen(signal)
        # (S5) Send feedback to listener
        listener.receive_feedback(concept_listener == concept)

        
    
    ### Methods used when agent listens

    def listen(self, signal):
        '''
         Agent listens to signal sent by speaker

         Args:
            signal: received signal
         
         Returns:
            concept_closest: concept which listener thinks is closest to heard signal
        '''
        # (L1) Receive signal
        # Save signal from speaker, used when updating
        signal_arr = signal.reshape(1,signal.shape[0])
        self.signal_received = signal
        # (L3) Find target closest to articulation
        # Save closest concept, used when updating later
        distances = cdist(self.language, signal_arr)
        self.concept_closest = np.argmin(distances)
        # (L4) Point to object
        # TODO: Is it strange that this function returns a value, while all other functions call a function on the other agent?
        #       Communication is implemented speaker-centred.
        return self.concept_closest
    
    def receive_feedback(self, feedback):
        '''
        Agent receives feedback from speaking agent, on correctness of concept,
        and updated its language table

        Args:
            feedback: True if and only if the object pointed to was correct
        '''
        # (L5) Update language table based on feedback
        if feedback:
            self.model.correct_interactions +=1
        print(f"Received signal: {self.signal_received}")
        signal_own = self.language[self.concept_closest]
        print(f"Closest own signal: {signal_own}")
        # Only if feedback positive, move own signal towards received
        # Every position in our signal is replaced by with a prob LEARNING_RATE
        # (implemented as boolean array)
        replace_positions = np.random.choice([True,False], size=N_FEATURES, p=[LEARNING_RATE,1-LEARNING_RATE])
        #random_vector = np.zeros(N_FEATURES)
        #signal_own.put(replace_positions, random_vector)
        if feedback:
            # If positive feedback: replace positions by received signal
            signal_own.put(replace_positions, self.signal_received)

            #zeros_vector = np.zeros(N_FEATURES)
            #signal_own.put(replace_positions, zeros_vector)
        else:
            # If negative feedback: replace positions by random signal
            a = self.signal_received
            signal_received_inv = np.where((a==0)|(a==1), a^1, a)
            signal_own.put(replace_positions, signal_received_inv)
            #random_vector = np.random.randint(0,2, N_FEATURES)
            #signal_own.put(replace_positions, random_vector)
        # NIET NODIG? self.language[self.concept_closest] = signal_own
        print(f"Own signal after update: {self.language[self.concept_closest]}")
        print()
        # After update, compute aggregate of articulation model, to color dot
        self.language_agg = self.compute_language_agg(self.language)



class Model(Model):
    '''
    Model class
    '''

    def __init__(self, height=HEIGHT, width=WIDTH, density=1.0, radius=MAX_RADIUS):
        '''
        Initialize field
        '''

        self.height = height
        self.width = width
        self.density = density
        self.radius = radius

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.global_model_distance = 0.0
        self.correct_interactions = 0.0
        self.steps = 0

        self.datacollector = DataCollector(
            {"global_model_distance": "global_model_distance",
            "proportion_correct_interactions": "proportion_correct_interactions"})

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            if np.random.rand() < self.density:
                agent = Agent((x, y), self)
                self.grid.position_agent(agent, (x, y))
                self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.steps+=1
        '''
        Run one step of the model.
        '''
        # Set correct interactions to 0 before all interactions are performed
        self.correct_interactions = 0.0
        self.schedule.step()
        # Now compute proportion of correct interaction
        self.proportion_correct_interactions = self.correct_interactions/float(N_AGENTS)
        if self.steps%1 == 0:
            self.global_model_distance = 0.0
            cumul_model_distance = 0
            n_pairs = 0
            # Compute test statistic by sampling some pairs
            agents = [a for a,x,y in self.grid.coord_iter()]
            agents_sample = np.random.choice(agents, SAMPLE , replace=False)
            agents1 = agents_sample[:HSAMPLE]
            agents2 = agents_sample[HSAMPLE:]
            for agent1 in agents1:
                for agent2 in agents2:
                    # Euclidean distance
                    dist = np.linalg.norm(agent1.language - agent2.language)
                    cumul_model_distance += dist
                    n_pairs +=1
            self.global_model_distance = float(cumul_model_distance)/float(n_pairs)

        self.datacollector.collect(self)
