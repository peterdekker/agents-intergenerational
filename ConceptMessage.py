class ConceptMessage:
    def __init__(self, lex_concept=None, person=None, transitivity=None):
        self.lex_concept = lex_concept
        self.person = person
        self.transitivity = transitivity
    

    def __str__(self):
        return f"{self.lex_concept}-{self.person}-{self.transitivity}"

    # Getters

    def get_lex_concept(self):
        return self.lex_concept

    def get_person(self):
        return self.person

    def get_transitivity(self):
        return self.transitivity

    # Setters

    def set_lex_concept(self, lex_concept):
        self.lex_concept

    def set_person(self, person):
        self.person = person

    def set_transitivity(self, transitivity):
        self.transitivity = transitivity

    # Compute loss, compared to other message

    def compute_success(self, other_message):
        lex_concept_sim = (self.lex_concept == other_message.lex_concept)
        person_sim = (self.person == other_message.person)
        transitivity_sim = (self.transitivity == other_message.transitivity)
        return lex_concept_sim and person_sim and transitivity_sim
