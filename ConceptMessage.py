class ConceptMessage:
    def __init__(self):
        self.lex_concept = None
        self.person = None
        self.transitivity = None

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
    

