from mesa import Agent
import copy
from collections import defaultdict
import numpy as np

from agents.signal import Signal
from agents.conceptmessage import ConceptMessage
from agents.config import RG, logging
from agents import misc


class Agent(Agent):
    def __init__(self, pos, model, data, init, capacity, l2):
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
        self.l2 = l2

        # These vars are not deep copies, because they will not be altered by agents
        # Always initialized from data, whatever init is.
        self.lex_concepts = data.lex_concepts
        self.persons = data.persons
        self.lex_concept_data = data.lex_concept_data

        # Only initialize affixes from data if init=='data'
        if init == "data":
            # Vars are deep copies from vars in data obj, so agent can change them
            # (deep copy because vars are nested dicts)
            self.affixes = copy.deepcopy(data.affixes)
        elif init == "empty":
            self.affixes = defaultdict(list)

    def step(self):
        '''
         Perform one interaction for this agent, with this agent as speaker
        '''
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
        concept_speaker, lex_concept, person, transitivity = ConceptMessage.draw_new_concept(self.lex_concepts,
                                                                                             self.persons,
                                                                                             self.lex_concept_data)
        logging.debug(f"Speaker chose concept: {concept_speaker!s}")

        form = self.lex_concept_data[lex_concept]["form"]
        signal.set_form(form)

        prefixing = self.lex_concept_data[lex_concept]["prefixing"]
        suffixing = self.lex_concept_data[lex_concept]["suffixing"]
        # (2) Based on verb and transitivity, add prefix or suffix:
        #  - prefixing verb:
        #     -- regardless of transitive or intransitive: use prefix
        prefixes = self.affixes[(lex_concept, person, "prefix")]
        if prefixing:
            prefix = ""
            # TODO: More elegant if len is always non-zero because there is always ""?
            if len(prefixes) > 0:
                prefix = RG.choice(prefixes)
                # Drop affix based on estimated intelligibility for listener (H&H)
                prefix = misc.reduce_affix_hh("prefixing", prefix, listener, self.model.reduction_hh)
                # Drop affix based on phonetic distance between stem/affix boundary phonemes
                prefix = misc.reduce_affix_phonetic("prefixing", prefix, form,
                                                    self.model.min_boundary_feature_dist)
            signal.set_prefix(prefix)

        #  - suffixing verb:
        #     -- transitive: do not use suffix
        #     -- intransitive: use suffix with probability, because it is not obligatory
        suffixes = self.affixes[(lex_concept, person, "suffix")]
        if suffixing:
            # In all cases where suffix will not be set, use empty suffix
            # (different from None, because listener will add empty suffix to its stack)
            suffix = ""
            # TODO: More elegant if len(sfxs) is always non-zero because there is always ""?
            if len(suffixes) > 0 and transitivity == "intrans":
                if RG.random() < self.model.suffix_prob:
                    suffix = RG.choice(suffixes)
                    suffix = misc.reduce_affix_hh("suffixing", suffix, listener, self.model.reduction_hh)
                    suffix = misc.reduce_affix_phonetic("suffixing", suffix, form,
                                                        self.model.min_boundary_feature_dist)
            signal.set_suffix(suffix)

        # (3) Add context from sentence (subject and object), based on transivitity.
        if RG.random() > self.model.drop_subject_prob:
            signal.set_subject(person)
        if transitivity == "trans":
            signal.set_object("OBJECT")


        # Send signal.
        logging.debug(f"Speaker sends signal: {signal!s}")
        concept_listener = listener.listen(signal)

        # Send feedback about correctness of concept to listener
        feedback = concept_speaker.compute_success(concept_listener)
        # if not feedback:
        #     if prefixing:
        #         print(f"Negative prefix: '{prefix}'")
        #     if suffixing:
        #         print(f"Negative suffix: '{suffix}'")
        listener.receive_feedback(feedback)

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
        lex_concept = misc.lookup_lex_concept(signal_form, self.lex_concepts, self.lex_concept_data)

        # Infer person from subject
        inferred_person = self.signal_recv.get_subject()
        if not inferred_person:
            # If person not inferred from context, try using affix
            inferred_person = misc.infer_person_from_signal(lex_concept,
                                                            self.lex_concept_data,
                                                            self.affixes,
                                                            self.persons,
                                                            signal)

        # We use directly existence/non-existence of object as criterion for transitivity
        transitivity = "trans" if self.signal_recv.get_object() else "intrans"

        self.concept_listener = ConceptMessage(
            lex_concept=lex_concept, person=inferred_person, transitivity=transitivity)
        logging.debug(f"Listener decodes concept: {self.concept_listener!s}")

        # Point to object
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
            # TODO: perform negative update on wrong affix as well?
            lex_concept_listener = self.concept_listener.get_lex_concept()
            person_listener = self.concept_listener.get_person()
            # Add current affix to right concept
            prefix_recv = self.signal_recv.get_prefix()
            suffix_recv = self.signal_recv.get_suffix()
            misc.update_affix_list("prefix", prefix_recv, self.affixes,
                                   self.lex_concept_data, lex_concept_listener, person_listener,
                                   self.capacity)
            misc.update_affix_list("suffix", suffix_recv, self.affixes,
                                   self.lex_concept_data, lex_concept_listener, person_listener,
                                   self.capacity)

    def is_l2(self):
        return self.l2
