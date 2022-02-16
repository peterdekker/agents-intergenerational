class Signal:
    def __init__(self, form=None, prefix=None, suffix=None, context_subject=None):
        self.form = form
        # Prefix and suffix are set to None by default
        # When a verb is prefixing/suffixing, prefix/suffix will get a string value
        # This string value can be "", which means empty affix. Difference with NOne
        # is that this will be added to listeners stack.
        self.prefix = prefix
        self.suffix = suffix
        self.subject = context_subject

    def __str__(self):
        return f"Form: {self.form} Prefix: {self.prefix} Suffix: {self.suffix} Subject: {self.subject}"
