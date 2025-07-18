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

# 6. Initial prompt and parser state
prompt = "fake-service open-file --file-name my-file.txt; fake-service "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
state = CLIParsingState(typestate=automaton.initial_state, automaton=automaton)

# First, get the state to parse the prompt
for char in prompt:
    state = state.parse_char(char)[0]

# Generation loop: apply appropriate masking at each step
max_steps = 10
api_name_prefix = []
constraining_api_name = False
expecting_separator = False  # Flag: after API name, only allow whitespace or semicolon
current_api_name = None  # Track the current API name for parameter masking
param_name_prefix = []  # Track the prefix of the parameter name being decoded

for step in range(max_steps):
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    decoded_so_far = tokenizer.decode(input_ids[0])
    print(f"\nStep {step}: parser_state.state = {state.state}, decoded_so_far = '{decoded_so_far[-50:]}'")

    # Decide which masking to apply
    if expecting_separator:
        print("\n~~~~~~GOT TO API NAME SEPARATOR MASKING~~~~~~\n")
        # Only allow whitespace, semicolon, or tokens that start with a space after API name
        # This ensures correct CLI structure: API name must be followed by a separator
        allowed = torch.zeros(logits.shape[-1], dtype=torch.bool)
        for token_id in range(logits.shape[-1]):
            token = tokenizer.decode([token_id])
            # Temporarily allow only whitespace
            if token == " ":
                allowed[token_id] = True
        allowed_tokens = [tokenizer.decode([i]) for i in range(logits.shape[-1]) if allowed[i]]
        print(f"[API Name Separator] Allowed tokens: {allowed_tokens[:10]}")
        mask = torch.full_like(logits, float('-inf'))
        mask[0, allowed] = 0
        filtered_logits = logits + mask
        next_token_id = torch.argmax(filtered_logits, dim=-1)
        next_token_str = tokenizer.decode(next_token_id)
        if next_token_str == " " or next_token_str == ";" or next_token_str == "--": #or next_token_str.startswith(" "):
            expecting_separator = False
        else:
            raise ValueError(f"Expected whitespace or semicolon, got {next_token_str}")
    elif state.state == "api" or constraining_api_name:
        print("GOT TO API NAME MASKING")
        # Dynamically get valid API names for the current typestate
        valid_api_names = list(automaton.transitions[state.typestate].keys()) if state.typestate is not None and state.typestate in automaton.transitions else []
        valid_api_name_token_ids = [tokenizer.encode(api, add_special_tokens=False) for api in valid_api_names]
        valid_api_name_trie = Trie()
        for api, ids in zip(valid_api_names, valid_api_name_token_ids):
            valid_api_name_trie.insert(ids, api)
        processor = TrieLogitsProcessor(valid_api_name_trie, api_name_prefix, tokenizer)
        filtered_logits = processor(input_ids, logits)
        next_token_id = torch.argmax(filtered_logits, dim=-1)
        api_name_prefix.append(next_token_id.item())
        # Check if we've completed an API name
        for api_ids, api in zip(valid_api_name_token_ids, valid_api_names):
            if api_name_prefix == api_ids:
                print(f"Completed API name: {tokenizer.decode(api_name_prefix)}")
                current_api_name = api  # <-- Track the current API name
                constraining_api_name = False
                api_name_prefix = []
                expecting_separator = True  # <-- Set flag to require whitespace or semicolon next
                break
        next_token_str = tokenizer.decode(next_token_id)
    elif state.state == "param_or_outfile":
        # Only allow parameter names valid for the current API
        allowed_param_names = parameter_names.get(current_api_name, []) if current_api_name is not None else []
        param_name_trie = Trie()
        param_name_token_ids = [tokenizer.encode(p, add_special_tokens=False) for p in allowed_param_names]
        for name, ids in zip(allowed_param_names, param_name_token_ids):
            param_name_trie.insert(ids, name)
        processor = TrieLogitsProcessor(param_name_trie, param_name_prefix, tokenizer)
        # processor = APINamesTrieLogitsProcessor(param_name_trie, param_name_prefix, tokenizer)
        filtered_logits = processor(input_ids, logits)
        next_token_id = torch.argmax(filtered_logits, dim=-1)
        param_name_prefix.append(next_token_id.item())
        # Check if we've completed a parameter name
        for param_ids in param_name_token_ids:
            if param_name_prefix == param_ids:
                print(f"Completed parameter name: {tokenizer.decode(param_name_prefix)}")
                param_name_prefix = []
                expecting_separator = True
                break
            else:
                print("DIDN'T COMPLETE PARAMETER NAME")
        next_token_str = tokenizer.decode(next_token_id)
    else:
        # CLI structure masking (semicolon/parameter enforcement)
        processor = CLIStructureLogitsProcessor(state, tokenizer)
        filtered_logits = processor(input_ids, logits)
        next_token_id = torch.argmax(filtered_logits, dim=-1)
        next_token_str = tokenizer.decode(next_token_id)

    print(f"Step {step}: '{next_token_str}'")
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
    state = state.parse_char(next_token_str)[0]

    # Encourage/require semicolon at the end of a command:
    # When parser expects a new command (state == 'service'), only allow 'fake-service' (or valid service names).
    # When parser expects a new API name (state == 'api'), only allow valid API names (Trie masking).
    # When parser expects a parameter or outfile (state == 'param_or_outfile'), only allow '--' or ';'.
    # After an API name, only allow whitespace or semicolon (see above logic).
    # This ensures that after a command, a semicolon is required to start the next command.

    if state.typestate == automaton.final_states[0]:
        print("Reached final typestate!")
        break

print("\nFinal output:", tokenizer.decode(input_ids[0]))
print("Parser state:", state.finalize())
