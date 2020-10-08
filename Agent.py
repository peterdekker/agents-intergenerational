from mesa import Agent

import copy
import numpy as np
from scipy.spatial.distance import cdist

import stats
import update
from Signal import Signal
from constants import RG


class Agent(Agent):
    def __init__(self, pos, model, data):
        '''
         Create a new speech agent.

         Args:
            pos: Agent initial location.
            model: Model in which agent is located.
        '''
        super().__init__(pos, model)
        self.pos = pos

        # These vars are not deep copies, because they will not be altered by agents(??)
        self.concepts = data.concepts
        self.persons = data.persons
        self.transitivities = data.transitivities

        # Vars are deep copies from vars in data obj, so agent can change them
        # (deep copy because vars are nested dicts)
        self.forms = copy.deepcopy(self.forms)
        self.model = copy.deepcopy(self.affixes)


        # self.language_agg = [0, 0, 0]

        # # Initialize array of concepts
        # self.concepts = np.arange(0, N_CONCEPTS)
        # # Initialize array of of language: draw binary features from uniform distribution
        # self.language = RG.randint(0, 2, (N_CONCEPTS, N_FEATURES))
        # self.language_agg = stats.compute_language_agg(self.language)


    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
        # Choose an agent to speak to
        # If density==1 and radius==MAX_RADIUS, every agent speaks with every other, so random mixing
        neighbors = self.model.grid.get_neighbors(self.pos, True, False, self.model.radius)
        listener = RG.choice(neighbors)
        self.speak(listener)

    # Methods used when agent speaks

    def speak(self, listener):
        '''
         Speak to other agent

         Args:
            listener: agent to speak to
        '''

        signal = Signal()

        # (1) Choose the meaning to be expressed, by picking a combination of:
        # 1. Lexical concept (row) (e.g. ala)
        # 2. Grammatical person (column) (e.g. 1SG)
        # 3. Transitive or intransitive (depending on allowed modes for this verb)
        concept = RG.choice(self.concepts)
        person = RG.choice(self.persons)
        transitivity = RG.choice(self.transitivities)

        # Use Lamaholot form, fall back to Alorese form
        if "lewoingu" in self.forms[concept]:
            forms = self.forms[concept]["lewoingu"]
        else:
            forms = self.forms[concept]["alorese"]
        form = RG.choice(forms.keys(), p=forms.values())
        signal.set_form(form)


        # (2) Based on verb and transitivity, add prefix or suffix:
        #  - prefixing verb:
        #     -- regardless of transitive or intransitive: use prefix
        prefixes = self.affixes[concept][person]["prefix"]
        if update.is_prefixing_verb(prefixes):
            prefix = RG.choice(prefixes.keys(), p=prefixes.values())
            signal.set_prefix(prefix)
        
        #  - suffixing verb:
        #     -- transitive: do not use prefix
        #     -- intransitive: use suffix with probability, because it is not obligatory
        suffixes = self.affixes[concept][person]["suffix"]
        if update.is_suffixing_verb(suffixes)):
            if transitivity == "intrans":
                if RG.random() < SUFFIX_PROB:
                    prefix = RG.choice(prefixes.keys(), p=prefixes.values())
                    signal.set_suffix(suffix)



        # If wordform is predictable enough (re-entrance),
        # and phonetic features at boundaries have high distance, do not add the affix with probability p.
        # TODO: to be implemented
        
        # (3) Add cues from sentence (Agent and Patient), based on transivitity. (pointers to objects)
        # TODO: Add possibility to drop pronoun probabilistically
        signal.set_context_subject(person)
        if transitivity == "trans":
            # TODO: Make this more finegrained?
            signal.set_context_object("OBJECT")

        # (4) Send signal.
        # 1. Adjust to listener? 
        # Noise?
        concept_listener = listener.listen(signal)

        # (5??) Send feedback to listener
        # listener.receive_feedback(concept_listener == concept)


        # signal = np.copy(self.language[concept])  # create copy, so noise not applied to original
        # if NOISE_RATE > 0.0:
        #     print("Apply noise")
        #     print(f"Signal before: {signal}")
        #     update.apply_noise(signal)
        #     print(f"Signal after: {signal}")
        

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