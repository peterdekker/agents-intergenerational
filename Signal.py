class Signal:
    def __init__(self):
        self.form = None
        self.prefix = None
        self.suffix = None
        self.context_subject = None
        self.context_object = None

    # Getters

    def get_form(self):
        return self.form

    def get_prefix(self):
        return self.prefix

    def get_suffix(self):
        return self.suffix
    
    def get_context_subject(self):
        return self.context_subject
    
    def get_context_object(self):
        return self.context_object
    
    # Setters

    def set_form(self, form):
        self.form = form

    def set_prefix(self, prefix):
        self.prefix = prefix
    
    def set_suffix(self, suffix):
        self.suffix = suffix

    def set_context_subject(self, context_subject):
        self.context_subject = context_subject
    
    def set_context_object(self, context_object):
        self.context_object = context_object
    

