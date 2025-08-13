#!/usr/bin/env python3

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from parser_api import CLIParsingState, EFSM, get_valid_transitions, get_allowed_values
from trie import Trie
import json
import os

class EFSMConstrainedLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that uses EFSM to constrain token generation.
    This is the key component for true constrained decoding.
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
        # Get current parser state
        current_state = self.state.typestate
        current_vars = self.state.internal_vars
        
        # Determine what we're currently generating based on parser state
        if self.state.state == "api":
            # We're generating an API name - constrain to valid transitions
            valid_transitions = get_valid_transitions(self.efsm, current_state, current_vars)
            valid_api_names = [t['symbol'] for t in valid_transitions]
            
            # Create mask for API names
            mask = self._create_api_name_mask(valid_api_names, input_ids, scores)
            
        elif self.state.state == "param_value" and self.state.current_param:
            # We're generating a parameter value - constrain based on predicates
            current_api = self.state.api_name
            allowed_values = get_allowed_values(self.efsm, current_state, current_api, current_vars)
            
            if allowed_values:
                # Constrain to allowed values from EFSM predicates
                mask = self._create_value_mask(allowed_values, input_ids, scores)
            else:
                # No specific constraints, allow any token
                mask = torch.zeros_like(scores)
                
        else:
            # Other states - allow any token
            mask = torch.zeros_like(scores)
            
        return scores + mask
    
    def _create_api_name_mask(self, valid_api_names, input_ids, scores):
        """Create mask to only allow valid API names."""
        mask = torch.full_like(scores, float('-inf'))
        
        # Build tries for valid API names
        api_tries = {}
        for api_name in valid_api_names:
            api_tokens = self.tokenizer.encode(api_name, add_special_tokens=False)
            api_tries[api_name] = api_tokens
            
        # Check each token position
        for token_id in range(scores.shape[-1]):
            # Check if this token could be part of any valid API name
            for api_name, api_tokens in api_tries.items():
                # Check if this token matches the next expected token in any API name
                current_prefix = input_ids[0].tolist()
                for i, expected_token in enumerate(api_tokens):
                    if i < len(current_prefix) and current_prefix[i] == expected_token:
                        continue
                    elif i == len(current_prefix) and token_id == expected_token:
                        # This token is valid for this API name
                        mask[0, token_id] = 0
                        break
                    else:
                        break
                        
        return mask
    
    def _create_value_mask(self, allowed_values, input_ids, scores):
        """Create mask to only allow specific parameter values."""
        mask = torch.full_like(scores, float('-inf'))
        
        # Build tries for allowed values
        value_tries = {}
        for value in allowed_values:
            value_tokens = self.tokenizer.encode(value, add_special_tokens=False)
            value_tries[value] = value_tokens
            
        # Check each token position
        for token_id in range(scores.shape[-1]):
            # Check if this token could be part of any allowed value
            for value, value_tokens in value_tries.items():
                current_prefix = input_ids[0].tolist()
                for i, expected_token in enumerate(value_tokens):
                    if i < len(current_prefix) and current_prefix[i] == expected_token:
                        continue
                    elif i == len(current_prefix) and token_id == expected_token:
                        # This token is valid for this value
                        mask[0, token_id] = 0
                        break
                    else:
                        break
                        
        return mask

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
                
                # Sample next token
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
                    
            # Check for completion
            if state.typestate in self.efsm.final_states:
                print("Reached final state!")
                break
                
        return self.tokenizer.decode(input_ids[0])

# Example usage
if __name__ == "__main__":
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
    
    # Load model
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create constrained generator
    generator = TrueConstrainedGenerator(efsm, parameter_names, tokenizer, model)
    
    # Generate with constraints
    prompt = "fake-service "
    result = generator.generate_constrained(prompt, max_length=20)
    print(f"\nFinal result: {result}") 