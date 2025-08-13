#!/usr/bin/env python3

import json
import os
from parser_api import EFSM, CLIParsingState, get_valid_transitions, get_allowed_values

script_dir = os.path.dirname(os.path.abspath(__file__))
automaton_json_path = os.path.join(script_dir, "typestates.json")

with open(automaton_json_path, "r") as f:
    automaton_data = json.load(f)

# Setup EFSM
efsm = EFSM(
    states=automaton_data["states"],
    symbols=automaton_data["symbols"],
    initial_state=automaton_data["initial_state"],
    final_states=automaton_data["final_states"],
    internal_vars=automaton_data.get("internal_vars", {}),
    transitions=automaton_data["transitions"]
)

print("=== EFSM Test ===")
print(f"EFSM: {efsm.states}")
print(f"Initial state: {efsm.initial_state}")
print(f"Internal vars: {efsm.internal_vars}")

# Test 1: Initial state
print("\n=== Test 1: Initial state ===")
current_state = "q1"
current_vars = {"f": None}
print(f"Current state: {current_state}, vars: {current_vars}")

valid_transitions = get_valid_transitions(efsm, current_state, current_vars)
print(f"Valid transitions: {[t['symbol'] for t in valid_transitions]}")

# Test 2: After opening a file
print("\n=== Test 2: After opening a file ===")
current_state = "q2"
current_vars = {"f": "my-file.txt"}
print(f"Current state: {current_state}, vars: {current_vars}")

valid_transitions = get_valid_transitions(efsm, current_state, current_vars)
print(f"Valid transitions: {[t['symbol'] for t in valid_transitions]}")

# Test allowed values for read-file
allowed_values = get_allowed_values(efsm, "q2", "read-file", current_vars)
print(f"Allowed values for read-file: {allowed_values}")

# Test 3: Parse a complete sequence
print("\n=== Test 3: Parse complete sequence ===")
test_sequence = "fake-service open-file --file-name my-file.txt; fake-service read-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt"

state = CLIParsingState(typestate=efsm.initial_state, efsm=efsm, internal_vars=efsm.internal_vars.copy())
print(f"Initial state: {state.typestate}, vars: {state.internal_vars}")

for char in test_sequence:
    state = state.parse_char(char)[0]
    if char == ";":
        print(f"After semicolon: state={state.typestate}, vars={state.internal_vars}")

result = state.finalize()
print(f"Final result: {result}")

# Test 4: Parse an invalid sequence
print("\n=== Test 4: Parse invalid sequence ===")
invalid_sequence = "fake-service read-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt"

state = CLIParsingState(typestate=efsm.initial_state, efsm=efsm, internal_vars=efsm.internal_vars.copy())
print(f"Initial state: {state.typestate}, vars: {state.internal_vars}")

for char in invalid_sequence:
    state = state.parse_char(char)[0]
    if char == ";":
        print(f"After semicolon: state={state.typestate}, vars={state.internal_vars}")

result = state.finalize()
print(f"Final result: {result}") 