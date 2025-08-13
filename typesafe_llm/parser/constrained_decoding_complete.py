#!/usr/bin/env python3

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from parser_api import CLIParsingState, EFSM, get_valid_transitions, get_allowed_values
import json
import os

class EFSMConstrainedLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that uses EFSM to constrain token generation.
    This actively prevents the LLM from generating invalid tokens.
    """
    
    def __init__(self, efsm, state, tokenizer, parameter_names):
        self.efsm = efsm
        self.state = state
        self.tokenizer = tokenizer
        self.parameter_names = parameter_names
        
    def __call__(self, input_ids, scores):
        """
        Mask logits based on current EFSM state to only allow valid next tokens.
        """
        # Create a mask that blocks all tokens by default
        mask = torch.full_like(scores, float('-inf'))
        
        # Get current parser state
        current_state = self.state.typestate
        current_vars = self.state.internal_vars
        
        print(f"Constraining generation - State: {self.state.state}, Typestate: {current_state}, Vars: {current_vars}")
        
        # Determine what we're currently generating based on parser state
        if self.state.state == "api":
            # We're generating an API name - constrain to valid transitions
            valid_transitions = get_valid_transitions(self.efsm, current_state, current_vars)
            valid_api_names = [t['symbol'] for t in valid_transitions]
            
            print(f"Valid API names: {valid_api_names}")
            
            # Allow tokens that could be part of valid API names
            for api_name in valid_api_names:
                api_tokens = self.tokenizer.encode(api_name, add_special_tokens=False)
                if api_tokens:
                    # Allow the first token of each valid API name
                    mask[0, api_tokens[0]] = 0
                    
        elif self.state.state == "param_value" and self.state.current_param:
            # We're generating a parameter value - constrain based on predicates
            current_api = self.state.api_name
            allowed_values = get_allowed_values(self.efsm, current_state, current_api, current_vars)
            
            print(f"Allowed values for {self.state.current_param}: {allowed_values}")
            
            if allowed_values:
                # Allow tokens that could be part of allowed values
                for value in allowed_values:
                    value_tokens = self.tokenizer.encode(value, add_special_tokens=False)
                    if value_tokens:
                        # Allow the first token of each allowed value
                        mask[0, value_tokens[0]] = 0
            else:
                # No specific constraints, allow common tokens
                common_tokens = self.tokenizer.encode("my-file.txt", add_special_tokens=False)
                for token_id in common_tokens:
                    mask[0, token_id] = 0
                    
        elif self.state.state == "param_or_outfile":
            # We're expecting a parameter name or separator
            if self.state.api_name:
                # Get all parameters for this API
                all_params = self.parameter_names.get(self.state.api_name, [])
                # Get already used parameters
                used_params = set(self.state.params.keys())
                # Get remaining unused parameters
                unused_params = [p for p in all_params if p not in used_params]
                
                print(f"All parameters for {self.state.api_name}: {all_params}")
                print(f"Used parameters: {used_params}")
                print(f"Unused parameters: {unused_params}")
                
                if unused_params:
                    # Allow unused parameter names
                    for param in unused_params:
                        param_tokens = self.tokenizer.encode(param, add_special_tokens=False)
                        if param_tokens:
                            mask[0, param_tokens[0]] = 0
                else:
                    # All parameters used, only allow separator
                    print("All parameters used, only allowing separator")
                    semicolon_token = self.tokenizer.encode(";", add_special_tokens=False)
                    if semicolon_token:
                        mask[0, semicolon_token[0]] = 0
                    
        else:
            # Other states - allow common tokens
            common_tokens = self.tokenizer.encode("open-file read-file write-file close-file --file-name", add_special_tokens=False)
            for token_id in common_tokens:
                mask[0, token_id] = 0
                
        return scores + mask

class TrueConstrainedGenerator:
    """
    A generator that uses EFSM to actively constrain LLM generation.
    This is the key difference - it doesn't just validate, it guides generation.
    """
    
    def __init__(self, efsm, parameter_names, tokenizer, model):
        self.efsm = efsm
        self.parameter_names = parameter_names
        self.tokenizer = tokenizer
        self.model = model
        
    def generate_constrained(self, prompt, max_length=50):
        """
        Generate text with EFSM constraints applied at each step.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # Initialize parser state
        state = CLIParsingState(
            typestate=self.efsm.initial_state,
            efsm=self.efsm,
            internal_vars=self.efsm.internal_vars.copy()
        )
        
        # Parse the prompt to get initial state
        for char in prompt:
            state = state.parse_char(char)[0]
            
        print(f"Initial state: {state.typestate}, vars: {state.internal_vars}")
        
        # Track current token being built
        current_token = ""
        
        # Generate with constraints
        for step in range(max_length):
            # Create constrained logits processor
            constrained_processor = EFSMConstrainedLogitsProcessor(
                self.efsm, state, self.tokenizer, self.parameter_names
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]
                
                # Apply constraints
                constrained_logits = constrained_processor(input_ids, logits)
                
                # Sample next token (greedy for demonstration)
                next_token_id = torch.argmax(constrained_logits, dim=-1)
                next_token = self.tokenizer.decode(next_token_id)
                
            print(f"Step {step}: Generated '{next_token}' (state: {state.state}, typestate: {state.typestate})")
            
            # Update input_ids
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            
            # Update parser state
            new_states = state.parse_char(next_token)
            if new_states:
                state = new_states[0]
                
                # Check if we completed an API call
                if next_token == ";":
                    print(f"Completed API call. New state: {state.typestate}, vars: {state.internal_vars}")
                    print(f"Parsed calls: {state.parsed_calls}")
                    
            # Check for completion
            if state.typestate in self.efsm.final_states:
                print("Reached final state!")
                break
                
        return self.tokenizer.decode(input_ids[0])

def demonstrate_parameter_tracking():
    """
    Demonstrate how parameter tracking works with a simple example.
    """
    print("=== Parameter Tracking Demonstration ===")
    
    # Load EFSM and parameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(script_dir, "typestates.json"), "r") as f:
        automaton_data = json.load(f)
        
    with open(os.path.join(script_dir, "parameter_names.json"), "r") as f:
        parameter_names = json.load(f)["fake-service"]
    
    # Create EFSM
    efsm = EFSM(
        states=automaton_data["states"],
        symbols=automaton_data["symbols"],
        initial_state=automaton_data["initial_state"],
        final_states=automaton_data["final_states"],
        internal_vars=automaton_data.get("internal_vars", {}),
        transitions=automaton_data["transitions"]
    )
    
    # Simulate parameter tracking for write-file
    print(f"\nParameters for write-file: {parameter_names['write-file']}")
    
    # Simulate different states of parameter usage
    test_cases = [
        {"api_name": "write-file", "used_params": set(), "description": "No parameters used"},
        {"api_name": "write-file", "used_params": {"--file-name"}, "description": "File name parameter used"},
    ]
    
    for case in test_cases:
        api_name = case["api_name"]
        used_params = case["used_params"]
        all_params = parameter_names.get(api_name, [])
        unused_params = [p for p in all_params if p not in used_params]
        
        print(f"\n{case['description']}:")
        print(f"  All parameters: {all_params}")
        print(f"  Used parameters: {used_params}")
        print(f"  Unused parameters: {unused_params}")
        
        if unused_params:
            print(f"  Would allow: {unused_params}")
        else:
            print(f"  Would only allow separator (;)")

# Example usage
if __name__ == "__main__":
    # Demonstrate parameter tracking
    demonstrate_parameter_tracking()
    
    print("\n" + "="*50)
    print("CONSTRAINED DECODING EXAMPLE")
    print("="*50)
    
    # Load model
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load EFSM and parameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(script_dir, "typestates.json"), "r") as f:
        automaton_data = json.load(f)
        
    with open(os.path.join(script_dir, "parameter_names.json"), "r") as f:
        parameter_names = json.load(f)["fake-service"]
    
    # Create EFSM
    efsm = EFSM(
        states=automaton_data["states"],
        symbols=automaton_data["symbols"],
        initial_state=automaton_data["initial_state"],
        final_states=automaton_data["final_states"],
        internal_vars=automaton_data.get("internal_vars", {}),
        transitions=automaton_data["transitions"]
    )
    
    # Create constrained generator
    generator = TrueConstrainedGenerator(efsm, parameter_names, tokenizer, model)
    
    # Generate with constraints
    prompt = "fake-service "
    result = generator.generate_constrained(prompt, max_length=30)
    print(f"\nFinal result: {result}") 