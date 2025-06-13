from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)

class TypeStateConstrainedLogitsProcessor(LogitsProcessor):
    """
    A logits processor that constrains the logits based on the type state.
    This processor is used to ensure that the model generates tokens that are
    consistent with the current type state.
    """

    def __init__(self, type_state):
        super().__init__()
        self.type_state = type_state
 
    def reset(self):
        self.reset_parser()
        self.reset_history()

    def reset_parser(self):
        """
        Reset the parser state.
        This method can be overridden to reset any parser-specific state.
        """
        pass
    
    def reset_history(self):
        self.history = []

    def process(self, inputs):
        pass
