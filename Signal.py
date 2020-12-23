class Signal:
    def __init__(self, form=None, prefix=None, suffix=None, context_subject=None, context_object=None):
        self.form = form
        # Prefix and suffix are set to None by default
        # When a verb is prefixing/suffixing, prefix/suffix will get a string value
        # This string value can be "", which means empty affix. Difference with NOne
        # is that this will be added to listeners stack.
        self.prefix = prefix
        self.suffix = suffix
        self.context_subject = context_subject
        self.context_object = context_object

    def __str__(self):
        return f"Form: {self.form} Prefix: {self.prefix} Suffix: {self.suffix} Subject: {self.context_subject} Object: {self.context_object}"

    # Getters

    def get_form(self):
        return self.form

    def get_prefix(self):
        return self.prefix

    def get_suffix(self):
        return self.suffix

    def get_subject(self):
        return self.context_subject

    def get_object(self):
        return self.context_object

    # Setters

    def set_form(self, form):
        self.form = form

    def set_prefix(self, prefix):
        self.prefix = prefix

    def set_suffix(self, suffix):
        self.suffix = suffix

    def set_subject(self, context_subject):
        self.context_subject = context_subject

    def set_object(self, context_object):
        self.context_object = context_object
    
    # Other

    def drop_affix(self):
        self.prefix = None
        self.suffix = None
