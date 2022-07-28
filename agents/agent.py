from mesa import Agent
import copy
from collections import defaultdict

from agents.signal import Signal
from agents.conceptmessage import ConceptMessage
from agents.config import RG, logging
from agents import misc
from agents import stats


class Agent(Agent):
    def __init__(self, pos, model, data, init, capacity, gen_production_old, gen_update_old, affix_prior, reduction_phonotactics, l2):
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
        self.gen_production_old = gen_production_old
        self.gen_update_old = gen_update_old
        self.affix_prior = affix_prior
        self.reduction_phonotactics = reduction_phonotactics
        self.l2 = l2

        # These vars are not deep copies, because they will not be altered by agents
        # Always initialized from data, whatever init is.
        self.lex_concepts = data.lex_concepts
        self.lex_concepts_type = data.lex_concepts_type
        self.persons = data.persons
        self.lex_concept_data = data.lex_concept_data

        # Only initialize affixes from data if init=='data'
        if init == "data":
            # Vars are deep copies from vars in data obj, so agent can change them
            # (deep copy because vars are nested dicts)
            self.affixes = copy.deepcopy(data.affixes)
        elif init == "empty":
            self.affixes = defaultdict(list)
        
        if self.model.browser_visualization:
            self.colours = stats.compute_colours(self)

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
        signal.form = form

        prefixing = self.lex_concept_data[lex_concept]["prefixing"]
        suffixing = self.lex_concept_data[lex_concept]["suffixing"]
        prefix = None
        suffix = None
        # (2) Based on verb and transitivity, add prefix or suffix:
        #  - prefixing verb:
        #     -- regardless of transitive or intransitive: use prefix
        if prefixing:
            # self.affixes[(lex_concept_gen, person_gen, "prefix")]
            prefixes = misc.retrieve_affixes_generalize(lex_concept, person, "prefix",
                                                        self.affixes, self.gen_production_old,
                                                        self.lex_concepts, self.persons, self.lex_concept_data)
            # TODO: More elegant if len is always non-zero because there is always ""?
            if len(prefixes) > 0:
                prefix = misc.affix_choice(prefixes)
                # # Drop affix based on estimated intelligibility for listener (H&H)
                # prefix = misc.reduce_hh("prefixing", prefix, listener, self.model.reduction_hh)
                # # Drop affix based on phonetic distance between stem/affix boundary phonemes
                # prefix = misc.reduce_boundary_feature_dist("prefixing", prefix, form,
                #                                                  self.model.min_boundary_feature_dist,
                #                                                  listener)
                prefix = misc.reduce_phonotactics("prefixing", prefix, form,
                                                  self.reduction_phonotactics, listener, self.model.clts)
            else:
                if self.model.send_empty_if_none:
                    prefix = ""
                else:
                    # Without option on: just skip this whole interaction. Only listen for this concept until it gets filled with at least one form.
                    return
            signal.prefix = prefix

        #  - suffixing verb:
        #     -- transitive: do not use suffix
        #     -- intransitive: use suffix with probability, because it is not obligatory
        if suffixing:
            # self.affixes[(lex_concept_gen, person_gen, "suffix")]
            suffixes = misc.retrieve_affixes_generalize(
                lex_concept, person, "suffix", self.affixes, self.gen_production_old,
                self.lex_concepts, self.persons, self.lex_concept_data)

            # In all cases where suffix will not be set, use empty suffix
            # (different from None, because listener will add empty suffix to its stack)
            # TODO: More elegant if len(sfxs) is always non-zero because there is always ""?
            if len(suffixes) > 0:
                if self.model.always_affix or transitivity == "intrans":
                    if self.model.always_affix or RG.random() < self.model.suffix_prob:
                        suffix = misc.affix_choice(suffixes)
                        # suffix = misc.reduce_hh("suffixing", suffix, listener, self.model.reduction_hh)
                        # suffix = misc.reduce_boundary_feature_dist("suffixing", suffix, form,
                        #                                                  self.model.min_boundary_feature_dist,
                        #                                                  listener)
                        suffix = misc.reduce_phonotactics("suffixing", suffix, form,
                                                          self.reduction_phonotactics, listener, self.model.clts)
            else:
                if self.model.send_empty_if_none:
                    suffix = ""
                else:
                    # Without option on: just skip this whole interaction. Only listen for this concept until it gets filled with at least one form.
                    return
            signal.suffix = suffix

        stats.update_communicated_model_stats(
            self.model, prefix, suffix, prefixing, suffixing, self.l2, self.model.steps)

        # (3) Add context from sentence (subject)
        if RG.random() >= self.model.pronoun_drop_prob:
            signal.subject = person

        # Send signal.
        logging.debug(f"Speaker sends signal: {signal!s}")
        concept_listener = listener.listen(signal)

        # Send feedback about correctness of concept to listener
        feedback = concept_speaker.compute_success(concept_listener)
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

        signal_form = self.signal_recv.form
        # Do reverse lookup in forms dict to find accompanying concept
        lex_concept_inferred = misc.lookup_lex_concept(signal_form, self.lex_concepts, self.lex_concept_data)

        # Infer person from subject
        person_inferred = self.signal_recv.subject
        if not person_inferred:
            # If person not inferred from context, try using affix
            person_inferred = misc.infer_person_from_signal(lex_concept_inferred,
                                                            self.lex_concept_data,
                                                            self.affixes,
                                                            self.persons,
                                                            signal)

        self.concept_listener = ConceptMessage(
            lex_concept=lex_concept_inferred, person=person_inferred)
        logging.debug(f"Listener decodes concept: {self.concept_listener!s}")

        # Point to inferred concept
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
            prefix_recv = self.signal_recv.prefix
            suffix_recv = self.signal_recv.suffix
            misc.update_affix_list(prefix_recv, suffix_recv, self.affixes, self.lex_concepts_type,
                                   self.lex_concept_data, self.persons, self.concept_listener,
                                   self.capacity, self.gen_update_old, self.l2)
        else:
            if self.model.negative_update:
                # Do negative update!
                prefix_recv = self.signal_recv.prefix
                suffix_recv = self.signal_recv.suffix
                misc.update_affix_list(prefix_recv, suffix_recv, self.affixes, self.lex_concepts_type,
                                       self.lex_concept_data, self.persons, self.concept_listener,
                                       self.capacity, self.gen_update_old, self.l2, negative=True)

    def is_l2(self):
        return self.l2
