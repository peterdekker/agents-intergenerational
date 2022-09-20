from agents.config import RG


class ConceptMessage:
    def __init__(self, lex_concept=None, person=None):
        self.lex_concept = lex_concept
        self.person = person

    @classmethod
    def draw_new_concept(concept_message_class, lex_concepts, persons, lex_concept_data):
        lex_concept = RG.choice(lex_concepts)
        person = RG.choice(persons)
        concept_message_instance = concept_message_class(lex_concept, person)
        return concept_message_instance, lex_concept, person

    def __str__(self):
        return f"{self.lex_concept}-{self.person}"

    # Compute loss, compared to other message

    def compute_success(self, other_message):
        lex_concept_sim = (self.lex_concept == other_message.lex_concept)
        person_sim = (self.person == other_message.person)
        return lex_concept_sim and person_sim
