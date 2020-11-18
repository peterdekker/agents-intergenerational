from mesa import Agent

import copy
import numpy as np

import stats
import util
import time
from Signal import Signal
from ConceptMessage import ConceptMessage
from constants import RG, SUFFIX_PROB, UPDATE_AMOUNT, logging


class Agent(Agent):
    def __init__(self, pos, model, init, data):
        '''
         Create a new speech agent.

         Args:
            pos: Agent initial location.
            model: Model in which agent is located.
            init: Initialization mode of forms and affixes: random or data
        '''
        super().__init__(pos, model)
        self.pos = pos

        # These vars are not deep copies, because they will not be altered by agents(??)
        # Always initialized from data, whatever init is.
        self.lex_concepts = data.lex_concepts
        self.persons = data.persons
        self.transitivities = data.transitivities

        # Only initialize forms and affixes from data if init==data
        if init=="data":
            # Vars are deep copies from vars in data obj, so agent can change them
            # (deep copy because vars are nested dicts)
            self.forms = copy.deepcopy(data.forms)
            self.affixes = copy.deepcopy(data.affixes)
        elif init=="empty":
            self.forms = defaultdict(dict)
            self.affixes = defaultdict(lambda: defaultdict(dict))


        self.colour = stats.compute_agent_colour(self.affixes)

    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
        # Choose an agent to speak to
        # If radius==MAX_RADIUS, every agent speaks with every other, so random mixing
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
        form = RG.choice(forms)
        #form = util.random_choice_weighted_dict(forms)
        signal.set_form(form)

        # (2) Based on verb and transitivity, add prefix or suffix:
        #  - prefixing verb:
        #     -- regardless of transitive or intransitive: use prefix
        prefixes = self.affixes[lex_concept][person]["prefix"]
        if util.is_prefixing_verb(prefixes):
            prefix = RG.choice(prefixes)
            #prefix = util.random_choice_weighted_dict(prefixes)
            signal.set_prefix(prefix)

        #  - suffixing verb:
        #     -- transitive: do not use prefix
        #     -- intransitive: use suffix with probability, because it is not obligatory
        suffixes = self.affixes[lex_concept][person]["suffix"]
        if util.is_suffixing_verb(suffixes):
            if transitivity == "intrans":
                if RG.random() < SUFFIX_PROB:
                    suffix = RG.choice(suffixes)
                    #suffix = util.random_choice_weighted_dict(suffixes)
                    signal.set_suffix(suffix)



        # If wordform is predictable enough (re-entrance),
        # and phonetic features at boundaries have high distance, do not add the affix with probability p.
        # TODO: to be implemented
        # TODO: implement phonetic conversion

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
        self.signal_recv = signal
        
        signal_form = self.signal_recv.get_form()
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
        person = self.signal_recv.get_context_subject()

        # We use directly existence/non-existence of object as criterion for transitivity
        transitivity = "trans" if self.signal_recv.get_context_object() else "intrans"
        
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
        logging.debug(f"Loss: {loss}")
        if loss==0.0:
            self.model.correct_interactions += 1
        
        # TODO: perform negative update on wrong prefix as well?
        # Update by target concept: the concept that was designated by the speaker
        lex_concept_speaker = concept_speaker.get_lex_concept()
        person_speaker = concept_speaker.get_person()
        # Add current prefix to right concept
        prefix_recv = self.signal_recv.get_prefix()
        suffix_recv = self.signal_recv.get_suffix()
        for affix_recv, affix_type in [(prefix_recv,"prefix"), (suffix_recv,"suffix")]:
            if affix_recv:
                self.affixes[lex_concept_speaker][person_speaker][affix_type].append(affix_recv)
                #affixes[affix_recv] += UPDATE_AMOUNT
                # Normalize probabilities
                #probs_sum = sum(affixes.values())
                #affixes = dict([(k,v/probs_sum) for k,v in affixes.items()])

        form_recv = self.signal_recv.get_form()
        self.forms[lex_concept_speaker]["form_lewoingu"].append(form_recv)

        # After update, compute aggregate of articulation model, to color dot
        self.colour=stats.compute_agent_colour(self.affixes)
