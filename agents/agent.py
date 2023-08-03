# from mesa import Agent
import copy
from collections import defaultdict

from agents.signal import Signal
from agents.conceptmessage import ConceptMessage
from agents.config import RG, logging
from agents import misc
from agents import stats


class Agent:
    def __init__(self, pos, model, data, init, affix_prior_combined, affix_prior, phonotactic_reduction, alpha, l2):
        '''
         Create a new speech agent.

         Args:
            pos: Agent initial location.
            model: Model in which agent is located.
            init: Initialization mode of forms and affixes: random or data
        '''
        self.model = model
        self.pos = pos
        self.affix_prior_combined = affix_prior_combined
        self.affix_prior = affix_prior
        self.phonotactic_reduction = phonotactic_reduction
        self.alpha = alpha
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

    def __str__(self):
        prop_prefix = stats.prop_internal_filled(self, "prefix")
        prop_suffix = stats.prop_internal_filled(self, "suffix")
        return f"ID:{self.pos}, l2: {self.l2}, prop_prefix: {prop_prefix}, prop_suffix: {prop_suffix}"

    def copy_parent(self, agents_prev):
        parent = RG.choice(agents_prev)
        # print(f"Agent {self.pos} learning from prev gen agent {parent.pos}")
        self.affixes = copy.deepcopy(parent.affixes)

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
        concept_speaker, lex_concept, person = ConceptMessage.draw_new_concept(self.lex_concepts,
                                                                               self.persons,
                                                                               self.lex_concept_data)

        logging.debug(f"Speaker chose concept: {concept_speaker!s}")

        form = self.lex_concept_data[lex_concept]["form"]
        signal.form = form

        prefixing = self.lex_concept_data[lex_concept]["prefix"]
        suffixing = self.lex_concept_data[lex_concept]["suffix"]
        prefix = None
        suffix = None
        if prefixing:
            if self.affix_prior_combined:

                # prefixes = list weighted by prob * prior_prob
                prefixes = misc.weighted_affixes_prior_combined(lex_concept, person, "prefix", self.affixes)
            elif self.affix_prior:
                prefixes = misc.use_affix_prior(lex_concept, person, "prefix", self.affixes, self.model.affix_prior_prob)
            else:
                prefixes = misc.distribution_from_exemplars(
                    lex_concept, person, "prefix", self.affixes, alpha=self.alpha)
            if len(prefixes) > 0:
                prefix = misc.affix_choice(prefixes)
                if self.phonotactic_reduction:
                    if RG.random() < self.model.phonotactic_reduction_prob:
                        prefix = misc.reduce_phonotactics(
                            "prefix", prefix, form, self.model.clts, self.model.cv_pattern_cache, drop_border_phoneme=self.model.phonotactic_reduction_drop_border_phoneme)
            else:
                # Just skip this whole interaction. Only listen for this concept until it gets filled with at least one form.
                return

            # if prefix == "":
            #     raise ValueError("Prefix is empty")
            signal.prefix = prefix

        #  - suffixing verb:
        #     -- transitive: do not use suffix
        #     -- intransitive: use suffix with probability, because it is not obligatory
        if suffixing:
            if self.affix_prior_combined:
                suffixes = misc.weighted_affixes_prior_combined(lex_concept, person, "suffix", self.affixes)
                # suffixes = list weighted by prob * prior_prob
            elif self.affix_prior:
                suffixes = misc.use_affix_prior(lex_concept, person, "suffix", self.affixes, self.model.affix_prior_prob)
            else:
                suffixes = misc.distribution_from_exemplars(
                    lex_concept, person, "suffix", self.affixes, alpha=self.alpha)
                # Do not use probabilities and prior probabilities of affixes in whole model.
                # Use plain exemplar lists. suffixes=unweighted list
                # suffixes = misc.retrieve_affixes_generalize(
                #     lex_concept, person, "suffix", self.affixes, self.gen_production_old)

            if len(suffixes) > 0:
                suffix = misc.affix_choice(suffixes)
                if self.phonotactic_reduction:
                    if RG.random() < self.model.phonotactic_reduction_prob:
                        suffix = misc.reduce_phonotactics(
                            "suffix", suffix, form, self.model.clts, self.model.cv_pattern_cache, drop_border_phoneme=self.model.phonotactic_reduction_drop_border_phoneme)
            else:
                # Just skip this whole interaction. Only listen for this concept until it gets filled with at least one form.
                return
            signal.suffix = suffix
        # stats.update_communicated_model_stats(
        #     self.model, prefix, suffix, prefixing, suffixing, self.l2)

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

        # Use affix to infer person
        person_inferred = misc.infer_person_from_signal(lex_concept_inferred,
                                                        self.lex_concept_data,
                                                        self.affixes,
                                                        self.persons,
                                                        signal)

        self.concept_listener = ConceptMessage(
            lex_concept=lex_concept_inferred, person=person_inferred)
        logging.debug(f"Listener decodes concept: {self.concept_listener!s}\n")

        # Point to inferred concept
        return self.concept_listener

    def receive_feedback(self, feedback_speaker):
        '''
        Listening agent receives concept meant by speaking agent,
        and updates its language table

        Args:
            feedback_speaker: feedback from the speaker
        '''
        self.model.total_interactions += 1
        if feedback_speaker:
            self.model.correct_interactions += 1
            prefix_recv = self.signal_recv.prefix
            suffix_recv = self.signal_recv.suffix
            misc.update_affix_list(prefix_recv, suffix_recv, self.affixes, self.lex_concepts_type,
                                   self.concept_listener)

    def is_l2(self):
        return self.l2
