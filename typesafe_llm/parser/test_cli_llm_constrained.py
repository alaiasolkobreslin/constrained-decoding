import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from parser_api import CLIParsingState, Automaton
from trie import Trie
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
automaton_json_path = os.path.join(script_dir, "typestates.json")
parameter_names_json_path = os.path.join(script_dir, "parameter_names.json")

with open(automaton_json_path, "r") as f:
    automaton_data = json.load(f)

with open(parameter_names_json_path, "r") as f:
    parameter_names = json.load(f)

# Setup automaton and parser
automaton = Automaton(
    states=automaton_data["states"],
    symbols=automaton_data["symbols"],
    transitions=automaton_data["transitions"],
    initial_state=automaton_data["initial_state"],
    final_states=automaton_data["final_states"]
)

# Load model and tokenizer
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Build a Trie of valid API names (tokenized)
api_names = ["open-file", "read-file", "write-file", "close-file"]
api_name_token_ids = [tokenizer.encode(api, add_special_tokens=False) for api in api_names]
api_name_trie = Trie()
for api, ids in zip(api_names, api_name_token_ids):
    api_name_trie.insert(ids, api)

class TrieLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, current_prefix_ids, tokenizer):
        self.trie = trie
        self.current_prefix_ids = current_prefix_ids  # List[int]
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        print("enter __call__ for TrieLogitsProcessor")
        allowed = torch.zeros(scores.shape[-1], dtype=torch.bool)
        for token_id in range(scores.shape[-1]):
            next_prefix = self.current_prefix_ids + [token_id]
            node = self.trie
            valid = True
            for tid in next_prefix:
                if tid in node._children:
                    node = node._children[tid]
                else:
                    valid = False
                    break
            if valid:
                allowed[token_id] = True
        allowed_tokens = [self.tokenizer.decode([i]) for i in range(scores.shape[-1]) if allowed[i]]
        print(f"[API Name] Allowed tokens: {allowed_tokens[:10]}")
        mask = torch.full_like(scores, float('-inf'))
        mask[0, allowed] = 0
        return scores + mask


class CLIStructureLogitsProcessor(LogitsProcessor):
    """
    This processor constrains the next token based on the parser state:
    - After an API name, only allow '--' (parameter), ';' (next command), or (optionally) a filename.
    - After '--', allow parameter names (for now, any alphanumeric token).
    - After a parameter value, only allow '--' (for another parameter) or ';' (to end the command and start a new one).
    - At the end of a command, a semicolon is strongly encouraged by only allowing ';' as the next token.

    Semicolon encouragement: By masking out all tokens except ';' at the end of a command (when parser expects a new command), we force the LLM to generate a semicolon, ensuring correct command separation and structure.
    """
    def __init__(self, parser_state, tokenizer):
        self.parser_state = parser_state
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        print(f"the corresponding tokens: {[tokenizer.decode(input_id) for input_id in input_ids]}")
        allowed = torch.zeros(scores.shape[-1], dtype=torch.bool)
        # Determine what is valid next, based on parser_state.state
        print(f"entering __call__ with parser_state.state == {self.parser_state.state}")
        if self.parser_state.state == "param_or_outfile":
            # Only allow '--' or ';' (or optionally a filename token)
            for token_id in range(scores.shape[-1]):
                token = self.tokenizer.decode([token_id])
                if token == "--" or token == ";":
                    allowed[token_id] = True
        elif self.parser_state.state == "param_value":
            # Allow any token (or restrict to alphanumeric)
            allowed[:] = True
        elif self.parser_state.state == "service":
            print("got to __call__ for CLIStructureLogitsProcessor, state == service")
            # Only allow 'fake-service' (or whatever service names you want)
            for token_id in range(scores.shape[-1]):
                token = self.tokenizer.decode([token_id])
                if token.strip() == "fake-service":
                    allowed[token_id] = True
        elif self.parser_state.state == "api":
            # TODO: fix this to allow only valid API names
            allowed[:] = True
        else:
            allowed[:] = True
        allowed_tokens = [self.tokenizer.decode([i]) for i in range(scores.shape[-1]) if allowed[i]]
        print(f"[CLI Structure] Allowed tokens: {allowed_tokens[:10]}")
        mask = torch.full_like(scores, float('-inf'))
        mask[0, allowed] = 0
        return scores + mask

class CLIConstrainedGenerator:
    """
    Generates CLI command sequences using typestate and parameter constraints.
    Handles token masking for API names, parameter names, separators, and CLI structure.
    """
    def __init__(self, automaton, parameter_names, tokenizer, model, max_steps=10):
        self.automaton = automaton
        self.parameter_names = parameter_names
        self.tokenizer = tokenizer
        self.model = model
        self.max_steps = max_steps

    def mask_with_trie(self, trie, prefix_ids, logits):
        """Return masked logits allowing only tokens that continue a valid path in the trie."""
        allowed = torch.zeros(logits.shape[-1], dtype=torch.bool)
        for token_id in range(logits.shape[-1]):
            next_prefix = prefix_ids + [token_id]
            node = trie
            valid = True
            for tid in next_prefix:
                if tid in node._children:
                    node = node._children[tid]
                else:
                    valid = False
                    break
            if valid:
                allowed[token_id] = True
        mask = torch.full_like(logits, float('-inf'))
        mask[0, allowed] = 0
        return logits + mask

    def mask_separator(self, logits):
        """Mask to only allow whitespace as separator after API/param name (can extend to ';' or '--')."""
        allowed = torch.zeros(logits.shape[-1], dtype=torch.bool)
        for token_id in range(logits.shape[-1]):
            token = self.tokenizer.decode([token_id])
            # if token == " " or token == ";" or token == "--":
            # TODO: fix this later to allow ;
            if token == " ":
                allowed[token_id] = True
        print(f"allowed: {[self.tokenizer.decode([token]) for token in range(logits.shape[-1]) if allowed[token]]}")
        mask = torch.full_like(logits, float('-inf'))
        mask[0, allowed] = 0
        return logits + mask

    def mask_cli_structure(self, state, logits):
        """Mask tokens based on CLI structure (e.g., only allow '--' or ';' in param_or_outfile)."""
        allowed = torch.zeros(logits.shape[-1], dtype=torch.bool)
        if state.state == "param_or_outfile":
            for token_id in range(logits.shape[-1]):
                token = self.tokenizer.decode([token_id])
                if token == "--" or token == ";":
                    allowed[token_id] = True
        else:
            allowed[:] = True
        mask = torch.full_like(logits, float('-inf'))
        mask[0, allowed] = 0
        return logits + mask

    def handle_separator(self, logits):
        """Handle the separator state (after API/param name)."""
        filtered_logits = self.mask_separator(logits)
        next_token_id = torch.argmax(filtered_logits, dim=-1)
        next_token_str = self.tokenizer.decode(next_token_id)
        if next_token_str == " " or next_token_str == ";":  # or next_token_str == "--":
            return next_token_id, next_token_str, False
        else:
            raise ValueError(f"Expected whitespace or semicolon, got {next_token_str}")

    def handle_api_name(self, state, api_name_prefix, logits):
        """Handle API name decoding with Trie masking."""
        valid_api_names = list(self.automaton.transitions[state.typestate].keys()) if state.typestate is not None and state.typestate in self.automaton.transitions else []
        valid_api_name_token_ids = [self.tokenizer.encode(api, add_special_tokens=False) for api in valid_api_names]
        valid_api_name_trie = Trie()
        for api, ids in zip(valid_api_names, valid_api_name_token_ids):
            valid_api_name_trie.insert(ids, api)
        filtered_logits = self.mask_with_trie(valid_api_name_trie, api_name_prefix, logits)
        next_token_id = torch.argmax(filtered_logits, dim=-1)
        api_name_prefix.append(next_token_id.item())
        current_api_name = None
        expecting_separator = False
        for api_ids, api in zip(valid_api_name_token_ids, valid_api_names):
            if api_name_prefix == api_ids:
                print(f"Completed API name: {self.tokenizer.decode(api_name_prefix)}")
                current_api_name = api
                api_name_prefix.clear()
                expecting_separator = True
                break
        next_token_str = self.tokenizer.decode(next_token_id)
        return next_token_id, next_token_str, current_api_name, expecting_separator

    def handle_param_name(self, current_api_name, param_name_prefix, logits):
        """Handle parameter name decoding with Trie masking for allowed params of current API."""
        allowed_param_names = self.parameter_names.get(current_api_name, []) if current_api_name is not None else []
        param_name_trie = Trie()
        param_name_token_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in allowed_param_names]
        for name, ids in zip(allowed_param_names, param_name_token_ids):
            param_name_trie.insert(ids, name)
        filtered_logits = self.mask_with_trie(param_name_trie, param_name_prefix, logits)
        next_token_id = torch.argmax(filtered_logits, dim=-1)
        param_name_prefix.append(next_token_id.item())
        expecting_separator = False
        for param_ids in param_name_token_ids:
            if param_name_prefix == param_ids:
                print(f"Completed parameter name: {self.tokenizer.decode(param_name_prefix)}")
                param_name_prefix.clear()
                expecting_separator = True
                break
        else:
            print("DIDN'T COMPLETE PARAMETER NAME")
        next_token_str = self.tokenizer.decode(next_token_id)
        return next_token_id, next_token_str, expecting_separator

    def handle_cli_structure(self, state, logits):
        """Handle CLI structure masking (e.g., param_or_outfile)."""
        filtered_logits = self.mask_cli_structure(state, logits)
        next_token_id = torch.argmax(filtered_logits, dim=-1)
        next_token_str = self.tokenizer.decode(next_token_id)
        return next_token_id, next_token_str

    def run(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        state = CLIParsingState(typestate=self.automaton.initial_state, automaton=self.automaton)
        for char in prompt:
            state = state.parse_char(char)[0]
        api_name_prefix = []
        param_name_prefix = []
        constraining_api_name = False
        expecting_separator = False
        current_api_name = None
        for step in range(self.max_steps):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            print(f"\nStep {step}: parser_state.state = {state.state}, decoded_so_far = '{self.tokenizer.decode(input_ids[0])[-50:]}'")
            if expecting_separator:
                next_token_id, next_token_str, expecting_separator = self.handle_separator(logits)
            elif state.state == "api" or constraining_api_name:
                next_token_id, next_token_str, new_api_name, new_expecting_separator = self.handle_api_name(state, api_name_prefix, logits)
                if new_api_name is not None:
                    current_api_name = new_api_name
                expecting_separator = new_expecting_separator
            elif state.state == "param_or_outfile":
                next_token_id, next_token_str, expecting_separator = self.handle_param_name(current_api_name, param_name_prefix, logits)
            else:
                next_token_id, next_token_str = self.handle_cli_structure(state, logits)
            print(f"Step {step}: '{next_token_str}'")
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            state = state.parse_char(next_token_str)[0]
            if state.typestate == self.automaton.final_states[0]:
                print("Reached final typestate!")
                break
        print("\nFinal output:", self.tokenizer.decode(input_ids[0]))
        print("Parser state:", state.finalize())

if __name__ == "__main__":
    prompt = "fake-service open-file --file-name my-file.txt; fake-service "
    generator = CLIConstrainedGenerator(automaton, parameter_names, tokenizer, model, max_steps=10)
    generator.run(prompt)
