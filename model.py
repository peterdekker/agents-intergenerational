from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit

from constants import HEIGHT, WIDTH, MAX_RADIUS, N_CONCEPTS, N_FEATURES, NOISE_STD, LEARNING_RATE, SAMPLE, HSAMPLE
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
        # Initialize array of of language: one uniform random feature array per concept
        self.language = np.random.rand(N_CONCEPTS, N_FEATURES)
        self.language_agg = self.compute_language_agg(self.language)
        print(self.language_agg)

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
        # (S0) Sample concept to talk about, from list of concepts
        concept = np.random.choice(self.concepts)

        # (S1+2) Sample new vocal tract position for this concept + add Gaussian noise
        articulation = self.language[concept]
        noise_art = np.random.normal(loc=0.0, scale=NOISE_STD, size=articulation.shape)
        articulation_noisy = articulation+noise_art

        # (S3) Generate sound (MFC) based on articulation parameters
        # + add Gaussian noise
        sound = articulation_noisy  # TODO: add LeVI here, now just identitify function
        noise_sound = np.random.normal(loc=0.0, scale=NOISE_STD, size=sound.shape)
        sound_noisy = sound + noise_sound

        # (S4) Send to listener, and receive concept listener points to
        concept_listener = listener.listen(sound_noisy)
        # (S5) Send feedback to listener
        self.send_feedback(concept_listener, concept, listener)
    
    def send_feedback(self, concept_listener, concept, listener):
        '''
         Agent sends feedback to listening agent just spoken to,
         on correctness of concept

         Args:
            concept_listener: the concept the listener returns, which it thinks the speaker spoke about
            concept: concept speaker actually spoke about
            listener: agent to give feedback, which this agent has just spoken to
        '''
        listener.receive_feedback(concept_listener != concept)
    
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
        # (L2) Perform inverse mapping, from sound to articulation that could have produced it
        articulation_inferred = signal  # TODO: add inverse mapping NN here, now just identity function
        articulation_inferred = articulation_inferred.reshape(1,articulation_inferred.shape[0])
        # Save inferred articulation from speaker, used when updating
        self.articulation_inferred_speaker = articulation_inferred
        # (L3) Find target closest to articulation
        distances = cdist(self.language, articulation_inferred)
        concept_closest = np.argmin(distances)
        # Save closest concept, used when updating later
        self.concept_closest = concept_closest
        # (L4) Point to object
        # TODO: Is it strange that this function returns a value, while all other functions call a function on the other agent?
        #       Communication is implemented speaker-centred.
        return concept_closest
    
    def receive_feedback(self, feedback):
        '''
        Agent receives feedback from speaking agent, on correctness of concept,
        and updated its articulation table

        Args:
            feedback: True if and only if the object pointed to was correct
        '''
        # (L5) Update articulation table based on feedback
        # Update becomes positive when feedback is True, negative when feedback is False
        sign = 1 if feedback else -1
        articulation_own = self.language[self.concept_closest]
        difference = self.articulation_inferred_speaker - articulation_own
        # Move own articulation towards or away from own articulation
        self.language[self.concept_closest] = articulation_own + sign * LEARNING_RATE * difference

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
        self.steps = 0

        self.datacollector = DataCollector(
            {"global_model_distance": "global_model_distance"})

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
        self.schedule.step()
        if self.steps%10 == 0:
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

            # Only collect data for graph every 10 steps
            self.datacollector.collect(self)
