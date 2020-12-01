from mesa import Agent

import copy
import util

import numpy as np

from Signal import Signal
from ConceptMessage import ConceptMessage
from constants import RG, logging
from collections import defaultdict
import editdistance


class Agent(Agent):
    def __init__(self, pos, model, data, init, capacity):
        '''
         Create a new speech agent.

         Args:
            pos: Agent initial location.
            model: Model in which agent is located.
            init: Initialization mode of forms and affixes: random or data
        '''
        super().__init__(pos, model)
        self.pos = pos
        self.capacity = capacity

        # These vars are not deep copies, because they will not be altered by agents(??)
        # Always initialized from data, whatever init is.
        self.lex_concepts = data.lex_concepts
        self.persons = data.persons
        self.transitivities = data.transitivities
        self.forms = data.forms

        # Only initialize affixes from data if init==data
        if init == "data":
            # Vars are deep copies from vars in data obj, so agent can change them
            # (deep copy because vars are nested dicts)
            self.affixes = copy.deepcopy(data.affixes)
        elif init == "empty":
            self.affixes = defaultdict(list)

        self.colour = self.compute_colour()

    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
        # Choose an agent to speak to
        # If radius==MAX_RADIUS, every agent speaks with every other, so random mixing
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

        # (1) Choose the concept to be expressed, by picking a combination of:
        # 1. Lexical concept (row) (e.g. ala)
        # 2. Grammatical person (column) (e.g. 1SG)
        # 3. Transitive or intransitive (depending on allowed modes for this verb)
        lex_concept = RG.choice(self.lex_concepts)
        person = RG.choice(self.persons)
        transitivity = RG.choice(self.transitivities[lex_concept])
        concept_speaker = ConceptMessage(lex_concept=lex_concept, person=person, transitivity=transitivity)
        logging.debug(f"Speaker chose concept: {concept_speaker!s}")

        form = self.forms[lex_concept]
        # TODO: Do something smarter than not speaking this iteration when there is no form
        # if len(forms) == 0:
        #     logging.debug("(L2) agent without forms for this concept, do not speak this interaction.")
        #     return
        signal.set_form(form)

        # (2) Based on verb and transitivity, add prefix or suffix:
        #  - prefixing verb:
        #     -- regardless of transitive or intransitive: use prefix
        prefixes = self.affixes[(lex_concept, person, "prefix")]
        if util.is_prefixing_verb(prefixes):
            prefix = RG.choice(prefixes)
            #prefix = util.random_choice_weighted_dict(prefixes)
            signal.set_prefix(prefix)

        #  - suffixing verb:
        #     -- transitive: do not use prefix
        #     -- intransitive: use suffix with probability, because it is not obligatory
        suffixes = self.affixes[(lex_concept, person, "suffix")]
        if util.is_suffixing_verb(suffixes):
            if transitivity == "intrans":
                if RG.random() < self.model.suffix_prob:
                    suffix = RG.choice(suffixes)
                    #suffix = util.random_choice_weighted_dict(suffixes)
                    signal.set_suffix(suffix)

        # If wordform is predictable enough (re-entrance),
        # and phonetic features at boundaries have high distance, do not add the affix with probability p.
        # TODO: to be implemented
        # TODO: implement phonetic conversion

        # (3) Add context from sentence (subject and object), based on transivitity.
        if RG.random() > self.model.drop_subject_prob:
            signal.set_subject(person)
        if transitivity == "trans":
            if RG.random() > self.model.drop_object_prob:
                signal.set_object("OBJECT")

        # Send signal.
        # TODO: noise?
        logging.debug(f"Speaker sends signal: {signal!s}")
        concept_listener = listener.listen(signal)

        # Send feedback about correctness of concept to listener
        feedback = concept_speaker.compute_success(concept_listener)
        if not feedback:
            print("FALSEE")
        listener.receive_feedback(feedback)

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
        inferred_lex_concept = None
        for lex_concept in self.forms:
            if self.forms[lex_concept] == signal_form:
                inferred_lex_concept = lex_concept
                break

        # TODO: take also prefix and suffix into consideration
        #possible_lex_concepts = []
        #lowest_dist = 1000
        # for lex_concept in self.forms:
        #     for internal_form in self.forms[lex_concept]:
        #         dist = editdistance.eval(signal_form, internal_form)
        #         if dist <= lowest_dist:
        #             if dist < lowest_dist:
        #                 lowest_dist = dist
        #                 # When dist of this form is strictly lower, empty list
        #                 possible_lex_concepts = []
        #             possible_lex_concepts.append(lex_concept)
        # RG.choice

        # We take directly person. TODO: do noisy comparison
        person = self.signal_recv.get_subject()

        # We use directly existence/non-existence of object as criterion for transitivity
        transitivity = "trans" if self.signal_recv.get_object() else "intrans"

        self.concept_listener = ConceptMessage(
            lex_concept=inferred_lex_concept, person=person, transitivity=transitivity)
        logging.debug(f"Listener decodes concept: {self.concept_listener!s}")
        # Point to object
        # TODO: Is it strange that this function returns a value, while all other functions call a function
        #       on the other agent? Communication is implemented speaker-centred.
        return self.concept_listener

    def receive_feedback(self, feedback_speaker):
        '''
        Listening agent receives concept meant by speaking agent,
        and updates its language table

        Args:
            feedback_speaker: feedback from the speaker
        '''

        if feedback_speaker:
            self.model.correct_interactions += 1
            # TODO: perform negative update on wrong prefix as well?
            # Update by target concept: the concept that was designated by the speaker
            lex_concept_listener = self.concept_listener.get_lex_concept()
            person_listener = self.concept_listener.get_person()
            # Add current prefix to right concept
            prefix_recv = self.signal_recv.get_prefix()
            suffix_recv = self.signal_recv.get_suffix()
            for affix_recv, affix_type in [(prefix_recv, "prefix"), (suffix_recv, "suffix")]:
                if affix_recv:
                    affix_list = self.affixes[(lex_concept_listener, person_listener, affix_type)]
                    affix_list.append(affix_recv)
                    logging.debug(
                        f"{affix_type.capitalize()}es after update: {affix_list}")
                    if len(affix_list) > self.capacity:
                        affix_list.pop(0)
                        logging.debug(
                            f"{affix_type.capitalize()}es longer than MAX, after drop: {affix_list}")

        # After update, compute aggregate of articulation model, to color dot
        self.colour = self.compute_colour()

    def morph_complexity(self):
        # TODO: optimize, get rid of loops
        lengths = []
        for lex_concept in self.lex_concepts:
            for person in self.persons:
                for affix_position in ["prefix", "suffix"]:
                    # Length is also calculated for empty affixes list (in L2 agents)
                    n_affixes = len(set(self.affixes[(lex_concept, person, affix_position)]))
                    lengths.append(n_affixes)
        mean_length = np.mean(lengths)  # if len(lengths)>0 else 0
        return mean_length

    def compute_colour(self):
        agg = self.morph_complexity() * 100
        return [250, 80, agg]
