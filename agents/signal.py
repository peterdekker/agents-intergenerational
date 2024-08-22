class Signal:
    def __init__(self, form=None, prefix=None, suffix=None):
        self.form = form
        # Prefix and suffix are set to None by default
        # When a verb is prefixing/suffixing, prefix/suffix will get a string value
        self.prefix = prefix
        self.suffix = suffix

    def __str__(self):
        return f"Form: {self.form} Prefix: {self.prefix} Suffix: {self.suffix}"
