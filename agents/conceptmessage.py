from agents.config import RG


class ConceptMessage:
    def __init__(self, lex_concept=None, person=None, transitivity=None):
        self.lex_concept = lex_concept
        self.person = person
        self.transitivity = transitivity

    @classmethod
    def draw_new_concept(concept_message_class, lex_concepts, persons, lex_concept_data):
        lex_concept = RG.choice(lex_concepts)
        person = RG.choice(persons)
        transitivity = RG.choice(lex_concept_data[lex_concept]["transitivities"])
        concept_message_instance = concept_message_class(lex_concept, person, transitivity)
        return concept_message_instance, lex_concept, person, transitivity

    def __str__(self):
        return f"{self.lex_concept}-{self.person}-{self.transitivity}"

    # Compute loss, compared to other message

    def compute_success(self, other_message):
        lex_concept_sim = (self.lex_concept == other_message.lex_concept)
        person_sim = (self.person == other_message.person)
        transitivity_sim = (self.transitivity == other_message.transitivity)
        return lex_concept_sim and person_sim and transitivity_sim
