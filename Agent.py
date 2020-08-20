from mesa import Agent

import numpy as np
from scipy.spatial.distance import cdist

from constants import N_CONCEPTS, N_FEATURES, NOISE_RATE
import stats
import update


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
        self.language = np.random.randint(0, 2, (N_CONCEPTS, N_FEATURES))
        self.language_agg = stats.compute_language_agg(self.language)

    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
        # Choose an agent to speak to
        # If density==1 and radius==MAX_RADIUS, every agent speaks with every other, so random mixing
        neighbors = self.model.grid.get_neighbors(self.pos, True, False, self.model.radius)
        listener = np.random.choice(neighbors)
        self.speak(listener)

    # Methods used when agent speaks

    def speak(self, listener):
        '''
         Speak to other agent

         Args:
            listener: agent to speak to
        '''
        concept = np.random.choice(self.concepts)

        signal = np.copy(self.language[concept])  # create copy, so noise not applied to original
        if NOISE_RATE > 0.0:
            print("Apply noise")
            print(f"Signal before: {signal}")
            update.apply_noise(signal)
            print(f"Signal after: {signal}")

        # (S4) Send to listener, and receive concept listener points to
        concept_listener = listener.listen(signal)
        # (S5) Send feedback to listener
        listener.receive_feedback(concept_listener == concept)

    # Methods used when agent listens

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
        signal_arr = signal.reshape(1, signal.shape[0])
        self.signal_received = signal
        # (L3) Find target closest to articulation
        # Save closest concept, used when updating later
        distances = cdist(self.language, signal_arr)
        self.concept_closest = np.argmin(distances)
        # (L4) Point to object
        # TODO: Is it strange that this function returns a value, while all other functions call a function
        #       on the other agent?
        #       Communication is implemented speaker-centred.
        return self.concept_closest

    def receive_feedback(self, feedback):
        '''
        Agent receives feedback from speaking agent, on correctness of concept,
        and updated its language table

        Args:
            feedback: True if and only if the object pointed to was correct
        '''
        if feedback:
            self.model.correct_interactions += 1
        print(f"Received signal: {self.signal_received}")
        signal_own = self.language[self.concept_closest]
        print(f"Closest own signal: {signal_own}")
        update.update_language(self.language, signal_own, self.signal_received, feedback)
        # NIET NODIG? self.language[self.concept_closest] = signal_own
        print(f"Own signal after update: {self.language[self.concept_closest]}")
        print()
        # After update, compute aggregate of articulation model, to color dot
        self.language_agg = stats.compute_language_agg(self.language)