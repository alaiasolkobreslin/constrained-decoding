import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LogitsProcessor

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

class DFA:
    def __init__(self):
        self.states = {"q0", "q1", "q2", "q3"}
        self.start_state = "q0"
        self.accept_states = {"q3"}
        self.transitions = {
            ("q0", "a"): "q1",
            ("q1", "b"): "q2",
            ("q2", "c"): "q3",
            ("q3", "d"): "q3",
        }

    def transition(self, state, char):
        return self.transitions.get((state, char), None)

    def reachable(self, state, visited=None):
        if visited is None:
            visited = set()
        if state in self.accept_states:
            return True
        visited.add(state)
        for (s, _), next_state in self.transitions.items():
            if s == state and next_state not in visited:
                if self.reachable(next_state, visited):
                    return True
        return False

dfa = DFA()

# LogitsProcessor to mask invalid transitions
class DFALogitsProcessor(LogitsProcessor):
    def __init__(self, current_state, dfa, tokenizer):
        self.state = current_state
        self.dfa = dfa
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        for token_id in range(scores.size(-1)):
            token = self.tokenizer.decode([token_id])
            if len(token) != 1 or not token.isalpha() or not token.islower():
                continue  # only allow single lowercase characters

            next_state = self.dfa.transition(self.state, token)
            if next_state and self.dfa.reachable(next_state):
                mask[0][token_id] = 0  # keep token

        return scores + mask

# Generation
prompt = "Start: "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
state = dfa.start_state
generated = tokenizer.decode(input_ids[0])

for _ in range(10):  # generate 10 characters
    logits = model(input_ids).logits[:, -1, :]
    processor = DFALogitsProcessor(state, dfa, tokenizer)
    filtered_logits = processor(input_ids, logits)
    probs = torch.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    char =tokenizer.decode(next_token.item())
    next_state = dfa.transition(state, char)
    if next_state is None:
        print(f"\nTerminated early: no transition for '{char}' from state {state}")
        break
    state = next_state

    input_ids = torch.cat([input_ids, next_token], dim=-1)
    generated += char

print(f"\nFinal output: {generated}")
print(f"Ended in state: {state}")
print(f"Accepting? {state in dfa.accept_states}")
