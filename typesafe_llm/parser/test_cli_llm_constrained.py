import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from parser_api import CLIParsingState, Automaton

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

# 3. Initial prompt and parser state
prompt = "fake-api open-file --file-name my-file.txt; fake-api "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
state = CLIParsingState(typestate=automaton.initial_state, automaton=automaton)

# 4. Generate step-by-step, masking out invalid tokens
max_steps = 20
for step in range(max_steps):
    # Get logits for next token
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)

    # Build mask: only allow tokens that keep parser alive
    allowed = torch.zeros(logits.shape[-1], dtype=torch.bool)
    for token_id in range(logits.shape[-1]):
        next_token = tokenizer.decode([token_id])
        # Only consider printable tokens (skip control chars, etc.)
        if not next_token.strip():
            continue
        # Try to parse the next token
        next_states = state.parse_char(next_token)
        if next_states:
            allowed[token_id] = True

    # If no allowed tokens, break
    if not allowed.any():
        print("No valid continuations!")
        break

    # Mask logits
    masked_logits = logits.masked_fill(~allowed.unsqueeze(0), -float("inf"))

    # Greedy decode (or sample)
    next_token_id = torch.argmax(masked_logits, dim=-1)
    next_token_str = tokenizer.decode(next_token_id)
    print(f"Step {step}: '{next_token_str}'")

    # Update input and parser state
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
    state = state.parse_char(next_token_str)[0]

    # Optionally, stop if parser is in accept/final state
    if state.typestate == automaton.final_states[0]:
        print("Reached final typestate!")
        break

print("Final output:", tokenizer.decode(input_ids[0]))
print("Parser state:", state.finalize()) 