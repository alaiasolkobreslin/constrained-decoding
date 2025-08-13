#!/usr/bin/env python3

import json
import os
from parser_api import CLIParsingState, EFSM

def demonstrate_parameter_tracking():
    """
    Demonstrate how parameter tracking prevents repetition and guides generation.
    """
    print("=== Parameter Tracking with Multiple Parameters ===")
    
    # Create a more complex parameter configuration
    complex_parameter_names = {
        "fake-service": {
            "open-file": ["--file-name", "--mode", "--encoding"],
            "write-file": ["--file-name", "--content", "--append"],
            "read-file": ["--file-name", "--start-line", "--end-line"],
            "close-file": ["--file-name"]
        }
    }
    
    # Simulate different states of parameter usage
    test_cases = [
        {
            "api_name": "write-file", 
            "used_params": set(), 
            "description": "No parameters used yet"
        },
        {
            "api_name": "write-file", 
            "used_params": {"--file-name"}, 
            "description": "File name parameter used"
        },
        {
            "api_name": "write-file", 
            "used_params": {"--file-name", "--content"}, 
            "description": "File name and content parameters used"
        },
        {
            "api_name": "write-file", 
            "used_params": {"--file-name", "--content", "--append"}, 
            "description": "All parameters used"
        },
    ]
    
    for case in test_cases:
        api_name = case["api_name"]
        used_params = case["used_params"]
        all_params = complex_parameter_names["fake-service"].get(api_name, [])
        unused_params = [p for p in all_params if p not in used_params]
        
        print(f"\n{case['description']}:")
        print(f"  API: {api_name}")
        print(f"  All parameters: {all_params}")
        print(f"  Used parameters: {used_params}")
        print(f"  Unused parameters: {unused_params}")
        
        if unused_params:
            print(f"  Would allow: {unused_params}")
        else:
            print(f"  Would only allow separator (;) - all parameters used!")
    
    print("\n" + "="*60)
    print("SIMULATED GENERATION SEQUENCE")
    print("="*60)
    
    # Simulate a generation sequence
    api_name = "write-file"
    all_params = complex_parameter_names["fake-service"][api_name]
    used_params = set()
    
    print(f"\nStarting generation for: {api_name}")
    print(f"Available parameters: {all_params}")
    
    # Simulate generating each parameter
    for param in all_params:
        used_params.add(param)
        unused_params = [p for p in all_params if p not in used_params]
        
        print(f"\nAfter using '{param}':")
        print(f"  Used: {used_params}")
        print(f"  Remaining: {unused_params}")
        
        if unused_params:
            print(f"  Next would allow: {unused_params}")
        else:
            print(f"  Next would only allow separator (;)")
    
    print(f"\nFinal state: All parameters used, ready for separator")

def demonstrate_efsm_integration():
    """
    Show how parameter tracking integrates with EFSM state.
    """
    print("\n" + "="*60)
    print("EFSM INTEGRATION WITH PARAMETER TRACKING")
    print("="*60)
    
    # Load EFSM
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
    
    # Simulate parser state with parameter tracking
    state = CLIParsingState(
        typestate="q2",  # File is open
        efsm=efsm,
        internal_vars={"f": "my-file.txt"}
    )
    
    # Simulate different stages of a write-file command
    stages = [
        {
            "stage": "Starting write-file command",
            "state": "api",
            "api_name": None,
            "params": {},
            "description": "About to generate API name"
        },
        {
            "stage": "API name generated",
            "state": "param_or_outfile", 
            "api_name": "write-file",
            "params": {},
            "description": "About to generate first parameter"
        },
        {
            "stage": "First parameter used",
            "state": "param_or_outfile",
            "api_name": "write-file", 
            "params": {"--file-name": "my-file.txt"},
            "description": "About to generate second parameter or separator"
        },
        {
            "stage": "All parameters used",
            "state": "param_or_outfile",
            "api_name": "write-file",
            "params": {"--file-name": "my-file.txt"},
            "description": "All parameters used, should only allow separator"
        }
    ]
    
    for stage in stages:
        print(f"\n{stage['stage']}:")
        print(f"  Parser state: {stage['state']}")
        print(f"  API name: {stage['api_name']}")
        print(f"  Used parameters: {stage['params']}")
        
        # Calculate what would be allowed
        if stage['api_name']:
            all_params = parameter_names.get(stage['api_name'], [])
            used_params = set(stage['params'].keys())
            unused_params = [p for p in all_params if p not in used_params]
            
            print(f"  All parameters: {all_params}")
            print(f"  Unused parameters: {unused_params}")
            
            if unused_params:
                print(f"  Would allow: {unused_params}")
            else:
                print(f"  Would only allow separator (;)")
        else:
            print(f"  Would allow API names: {efsm.symbols}")
        
        print(f"  Description: {stage['description']}")

if __name__ == "__main__":
    demonstrate_parameter_tracking()
    demonstrate_efsm_integration() 