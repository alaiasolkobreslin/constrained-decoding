import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from parser_api import CLIParsingState, Automaton
from trie import Trie

# 1. Setup: Automaton and parser
automaton = Automaton(
    states=["start", "opened", "closed"],
    symbols=["open-file", "read-file", "write-file", "close-file"],
    transitions={
        "start": {"open-file": "opened"},
        "opened": {"read-file": "opened", "write-file": "opened", "close-file": "closed"},
        "closed": {}
    },
    initial_state="start",
    final_states=["closed"]
)

# 2. Load model and tokenizer (use a small model for testing)
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Build a Trie of valid API names (tokenized)
api_names = ["open-file", "read-file", "write-file", "close-file"]
api_name_token_ids = [tokenizer.encode(api, add_special_tokens=False) for api in api_names]
print(f"api_name_token_ids: {api_name_token_ids}")
api_name_trie = Trie()
for api, ids in zip(api_names, api_name_token_ids):
    api_name_trie.insert(ids, api)

# 4. Custom LogitsProcessor for API name masking (Trie-based)
class APINamesTrieLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, current_prefix_ids, tokenizer):
        self.trie = trie
        self.current_prefix_ids = current_prefix_ids  # List[int]
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        print("enter __call__ for APINamesTrieLogitsProcessor")
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
        print (f"allowed: {allowed}")
        allowed_tokens = [self.tokenizer.decode([i]) for i in range(scores.shape[-1]) if allowed[i]]
        print(f"[API Name] Allowed tokens: {allowed_tokens}")
        mask = torch.full_like(scores, float('-inf'))
        mask[0, allowed] = 0
        return scores + mask

# 5. Custom LogitsProcessor for CLI structure (semicolon/parameter enforcement)
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
        allowed = torch.zeros(scores.shape[-1], dtype=torch.bool)
        # Determine what is valid next, based on parser_state.state
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
            # Only allow 'fake-api' (or whatever service names you want)
            for token_id in range(scores.shape[-1]):
                token = self.tokenizer.decode([token_id])
                # print(f"token.strip(): {token.strip()}")
                if token.strip() == "fake-api":
                    # print(f"allowed token HERE: {token_id}")
                    allowed[token_id] = True
        elif self.parser_state.state == "api":
            # This state is handled by the Trie-based processor
            # TODO: should we really allow all tokens here?
            allowed[:] = True
        else:
            # Default: allow all
            # TODO: should the default be to allow all tokens?
            allowed[:] = True
        allowed_tokens = [self.tokenizer.decode([i]) for i in range(scores.shape[-1]) if allowed[i]]
        print(f"[CLI Structure] Allowed tokens: {allowed_tokens}")
        mask = torch.full_like(scores, float('-inf'))
        mask[0, allowed] = 0
        return scores + mask

# 6. Initial prompt and parser state
prompt = "fake-api open-file --file-name my-file.txt; fake-api "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
state = CLIParsingState(typestate=automaton.initial_state, automaton=automaton)

# 7. Generation loop: apply appropriate masking at each step
max_steps = 40
api_name_prefix = []
constraining_api_name = False
for step in range(max_steps):
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    decoded_so_far = tokenizer.decode(input_ids[0])
    print(f"\nStep {step}: parser_state.state = {state.state}, decoded_so_far = '{decoded_so_far[-50:]}'")

    # Decide which masking to apply
    if state.state == "api" or constraining_api_name:
        print("GOT TO API NAME MASKING")
        # Trie-based API name masking
        constraining_api_name = True
        processor = APINamesTrieLogitsProcessor(api_name_trie, api_name_prefix, tokenizer)
        filtered_logits = processor(input_ids, logits)
        next_token_id = torch.argmax(filtered_logits, dim=-1)
        api_name_prefix.append(next_token_id.item())
        # Check if we've completed an API name
        for api_ids in api_name_token_ids:
            if api_name_prefix == api_ids:
                print(f"Completed API name: {tokenizer.decode(api_name_prefix)}")
                constraining_api_name = False
                api_name_prefix = []
                break
    else:
        print("DIDN'T GET TO API NAME MASKING :(")
        # CLI structure masking (semicolon/parameter enforcement)
        processor = CLIStructureLogitsProcessor(state, tokenizer)
        filtered_logits = processor(input_ids, logits)
        print(f"filtered_logits: {filtered_logits}")
        print(f"filtered_logits.shape: {filtered_logits.shape}")
        print(f"torch.max(filtered_logits): {torch.max(filtered_logits)}")
        next_token_id = torch.argmax(filtered_logits, dim=-1)

    next_token_str = tokenizer.decode(next_token_id)
    print(f"Step {step}: '{next_token_str}'")
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
    state = state.parse_char(next_token_str)[0]

    # Encourage/require semicolon at the end of a command:
    # When parser expects a new command (state == 'service'), only allow 'fake-api' (or valid service names).
    # When parser expects a new API name (state == 'api'), only allow valid API names (Trie masking).
    # When parser expects a parameter or outfile (state == 'param_or_outfile'), only allow '--' or ';'.
    # This ensures that after a command, a semicolon is required to start the next command.

    if state.typestate == automaton.final_states[0]:
        print("Reached final typestate!")
        break

print("\nFinal output:", tokenizer.decode(input_ids[0]))
print("Parser state:", state.finalize())