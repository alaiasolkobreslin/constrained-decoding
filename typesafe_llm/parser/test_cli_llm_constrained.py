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
api_name_trie = Trie()
for api, ids in zip(api_names, api_name_token_ids):
    api_name_trie.insert(ids, api)

# 4. Custom LogitsProcessor for API name masking
class APINamesTrieLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, current_prefix_ids):
        self.trie = trie
        self.current_prefix_ids = current_prefix_ids  # List[int]

    def __call__(self, input_ids, scores):
        # Only allow tokens that could continue a valid API name from the current prefix
        allowed = torch.zeros(scores.shape[-1], dtype=torch.bool)
        for token_id in range(scores.shape[-1]):
            next_prefix = self.current_prefix_ids + [token_id]
            # Check if this prefix exists in the trie
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
        # Print allowed tokens for debugging
        allowed_tokens = [tokenizer.decode([i]) for i in range(scores.shape[-1]) if allowed[i]]
        print(f"Allowed tokens: {allowed_tokens}")
        mask = torch.full_like(scores, float('-inf'))
        mask[0, allowed] = 0
        return scores + mask

# 5. Initial prompt and parser state
prompt = "fake-api open-file --file-name my-file.txt; fake-api "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
state = CLIParsingState(typestate=automaton.initial_state, automaton=automaton)

# 6. Generation loop: only constrain at API name position
max_steps = 20
api_name_prefix = []
constraining = False
for step in range(max_steps):
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]

    # Determine if we are at the point of parsing an API name
    # (after 'fake-api ' or after a semicolon)
    decoded_so_far = tokenizer.decode(input_ids[0])
    if decoded_so_far.endswith("fake-api ") or constraining:
        constraining = True
        processor = APINamesTrieLogitsProcessor(api_name_trie, api_name_prefix)
        filtered_logits = processor(input_ids, logits)
        # Greedy decode
        next_token_id = torch.argmax(filtered_logits, dim=-1)
        api_name_prefix.append(next_token_id.item())
        # Check if we've completed an API name
        for api_ids in api_name_token_ids:
            if api_name_prefix == api_ids:
                print(f"Completed API name: {tokenizer.decode(api_name_prefix)}")
                constraining = False
                api_name_prefix = []
                break
    else:
        next_token_id = torch.argmax(logits, dim=-1)

    next_token_str = tokenizer.decode(next_token_id)
    print(f"Step {step}: '{next_token_str}'")
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
    state = state.parse_char(next_token_str)[0]

    if state.typestate == automaton.final_states[0]:
        print("Reached final typestate!")
        break

print("Final output:", tokenizer.decode(input_ids[0]))
print("Parser state:", state.finalize())