class Signal:
    def __init__(self, form=None, prefix=None, suffix=None, context_subject=None, context_object=None):
        self.form = form
        self.prefix = prefix
        self.suffix = suffix
        self.context_subject = context_subject
        self.context_object = context_object

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
    

