from mesa import Agent

import copy
import numpy as np

import stats
import update
import util
import time
from Signal import Signal
from ConceptMessage import ConceptMessage
from constants import RG, SUFFIX_PROB, logging


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
        self.lex_concepts = data.lex_concepts
        self.persons = data.persons
        self.transitivities = data.transitivities

        # Vars are deep copies from vars in data obj, so agent can change them
        # (deep copy because vars are nested dicts)
        self.forms = copy.deepcopy(data.forms)
        self.affixes = copy.deepcopy(data.affixes)

        # # Initialize array of concepts
        # self.concepts = np.arange(0, N_CONCEPTS)
        # # Initialize array of of language: draw binary features from uniform distribution
        # self.language = RG.randint(0, 2, (N_CONCEPTS, N_FEATURES))
        self.colour = stats.compute_agent_colour(self.affixes)

    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
        # Choose an agent to speak to
        # If density==1 and radius==MAX_RADIUS, every agent speaks with every other, so random mixing
        neighbors = self.model.grid.get_neighbors(self.pos, True, False, self.model.radius)
        listener = RG.choice(neighbors)
        #start_speak = time.time()
        self.speak(listener)
        #end_speak = time.time()
        #print(f"speak:{end_speak-start_speak}")

    # Methods used when agent speaks

    def speak(self, listener):
        '''
         Speak to other agent

         Args:
            listener: agent to speak to
        '''

        signal = Signal()

        # (1) Choose the concept to be expressed, by picking a combination of:
        # 1. Lexical concept (row) (e.g. ala)
        # 2. Grammatical person (column) (e.g. 1SG)
        # 3. Transitive or intransitive (depending on allowed modes for this verb)
        lex_concept = RG.choice(self.lex_concepts)
        person = RG.choice(self.persons)
        transitivity = RG.choice(self.transitivities[lex_concept])
        concept_speaker = ConceptMessage(lex_concept=lex_concept, person=person, transitivity=transitivity)
        logging.debug(f"Speaker chose concept: {concept_speaker!s}")

        # Use Lewoingu Lamaholot form, fall back to Alorese form
        forms = self.forms[lex_concept]["lewoingu"]
        # TODO: Choosing between forms has not much effect, form probabilities are not yet updated
        form = util.random_choice_weighted_dict(forms)
        signal.set_form(form)

        # (2) Based on verb and transitivity, add prefix or suffix:
        #  - prefixing verb:
        #     -- regardless of transitive or intransitive: use prefix
        prefixes = self.affixes[lex_concept][person]["prefix"]
        if util.is_prefixing_verb(prefixes):
            prefix = util.random_choice_weighted_dict(prefixes)
            signal.set_prefix(prefix)

        #  - suffixing verb:
        #     -- transitive: do not use prefix
        #     -- intransitive: use suffix with probability, because it is not obligatory
        suffixes = self.affixes[lex_concept][person]["suffix"]
        if util.is_suffixing_verb(suffixes):
            if transitivity == "intrans":
                if RG.random() < SUFFIX_PROB:
                    suffix = util.random_choice_weighted_dict(suffixes)
                    signal.set_suffix(suffix)



        # If wordform is predictable enough (re-entrance),
        # and phonetic features at boundaries have high distance, do not add the affix with probability p.
        # TODO: to be implemented

        # (3) Add context from sentence (subject and object), based on transivitity.
        # TODO: Add possibility to drop pronoun probabilistically
        signal.set_context_subject(person)
        if transitivity == "trans":
            # TODO: Make this more finegrained?
            signal.set_context_object("OBJECT")


        # Send signal.
        # TODO: noise? 
        logging.debug(f"Speaker sends signal: {signal!s}")
        concept_listener=listener.listen(signal)

        # Send real concept as feedback to listener
        # TODO: experiment with only sending correct/incorrect
        listener.receive_feedback(concept_speaker)

        # Only listener updates TODO: also experiment with speaker updating


    # Methods used when agent listens

    def listen(self, signal):
        '''
         Agent listens to signal sent by speaker

         Args:
            signal: received signal

         Returns:
            message: concept which listener thinks is closest to heard signal
        '''
        
        signal_form = signal.get_form()
        # Do reverse lookup in forms dict to find accompanying concept
        # TODO: Maybe create reverse dict beforehand to speed up
        # TODO: noisy comparison
        inferred_lex_concept = None
        for lex_concept in self.forms:
            if signal_form in self.forms[lex_concept]['lewoingu']:
                inferred_lex_concept = lex_concept
                break

        # TODO: take also prefix and suffix into consideration

        # We take directly person. TODO: do noisy comparison
        person = signal.get_context_subject()

        # We use directly existence/non-existence of object as criterion for transitivity
        if signal.get_context_object():
            transitivity = "trans"
        else:
            transitivity = "intrans"
        
        self.concept_listener = ConceptMessage(lex_concept=inferred_lex_concept, person=person, transitivity=transitivity)
        logging.debug(f"Listener decodes concept: {self.concept_listener!s}")
        # Point to object
        # TODO: Is it strange that this function returns a value, while all other functions call a function
        #       on the other agent? Communication is implemented speaker-centred.
        return self.concept_listener

    def receive_feedback(self, concept_speaker):
        '''
        Listening agent receives concept meant by speaking agent, 
        and updates its language table

        Args:
            concept_speaker: concept intended by speaker
        '''

        loss = self.concept_listener.compute_loss(concept_speaker)
        logging.debug(f"Loss: {loss}\n")
        # TODO: perform update

        if loss==0:
            self.model.correct_interactions += 1
        #signal_own=self.language[self.concept_closest]
        #update.update_language(self.language, signal_own, self.signal_received, feedback)
        # NIET NODIG? self.language[self.concept_closest] = signal_own

        # After update, compute aggregate of articulation model, to color dot
        #self.colour=stats.compute_agent_colour(self.affixes)
