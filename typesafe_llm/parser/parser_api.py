from typing import Dict, Optional, List, Any, Tuple, Set
from parser_base import IncrementalParsingState
import dataclasses

# Extended Finite State Machine for typestate tracking
class EFSM:
    def __init__(self, 
                 states: List[str], 
                 symbols: List[str], 
                 initial_state: str, 
                 final_states: List[str],
                 internal_vars: Dict[str, Any] = None,
                 transitions: List[Dict] = None):
        """
        Extended Finite State Machine
        
        Args:
            states: List of state names
            symbols: List of possible API names/symbols
            initial_state: Starting state
            final_states: List of accepting states
            internal_vars: Initial values for internal state variables
            transitions: List of transition rules, each with:
                - from_state: source state
                - symbol: API name/symbol
                - predicate: dict of equality conditions on parameters
                - to_state: target state
                - var_updates: dict of internal variable updates
        """
        self.states = states
        self.symbols = symbols
        self.initial_state = initial_state
        self.final_states = final_states
        self.internal_vars = internal_vars or {}
        self.transitions = transitions or []
        
    def transition(self, current_state: str, symbol: str, params: Dict[str, str], current_vars: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Compute the next state and updated internal variables based on current state, symbol, and parameters.
        
        Returns:
            Tuple of (next_state, updated_vars) or None if no valid transition
        """
        for transition in self.transitions:
            if (transition['from_state'] == current_state and 
                transition['symbol'] == symbol):
                
                # Check if predicate is satisfied
                if self._check_predicate(transition.get('predicate', {}), params, current_vars):
                    # Apply variable updates
                    new_vars = current_vars.copy()
                    for var_name, update_expr in transition.get('var_updates', {}).items():
                        new_vars[var_name] = self._evaluate_update(update_expr, params, current_vars)
                    
                    return transition['to_state'], new_vars
        
        return None
    
    def _check_predicate(self, predicate: Dict[str, str], params: Dict[str, str], current_vars: Dict[str, Any]) -> bool:
        """
        Check if a predicate (conjunction of equality statements) is satisfied.
        
        Args:
            predicate: Dict mapping parameter names to expected values (can reference internal vars)
            params: Actual parameter values from the API call
            current_vars: Current internal variable values
            
        Returns:
            True if all equality conditions are satisfied
        """
        for param_name, expected_value in predicate.items():
            # If expected_value is a string that references an internal variable
            if isinstance(expected_value, str) and expected_value in current_vars:
                actual_value = current_vars[expected_value]
            else:
                actual_value = expected_value
            
            if params.get(param_name) != actual_value:
                return False
        return True
    
    def _evaluate_update(self, update_expr: Any, params: Dict[str, str], current_vars: Dict[str, Any]) -> Any:
        """
        Evaluate an update expression for internal variables.
        
        Args:
            update_expr: The update expression (can be a value, param reference, or function)
            params: Current parameter values
            current_vars: Current internal variable values
            
        Returns:
            The computed new value
        """
        if isinstance(update_expr, str):
            # If it's a string, it could be a parameter reference
            if update_expr in params:
                return params[update_expr]
            elif update_expr in current_vars:
                return current_vars[update_expr]
            else:
                return update_expr
        elif callable(update_expr):
            # If it's a function, call it with params and current_vars
            return update_expr(params, current_vars)
        else:
            # Otherwise, return the value as-is
            return update_expr

    def is_final(self, state: str) -> bool:
        return state in self.final_states

class CLIParsingState(IncrementalParsingState):
    def __init__(
        self,
        current_token: str = "",
        state: str = "service",
        service: Optional[str] = None,
        api_name: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
        current_param: Optional[str] = None,
        outfile: Optional[str] = None,
        accept: bool = False,
        # Typestate tracking with EFSM
        typestate: Optional[str] = None,
        efsm: Optional[EFSM] = None,
        internal_vars: Optional[Dict[str, Any]] = None,
        parsed_calls: Optional[List[Dict[str, Any]]] = None,
        # Track opened files globally
        opened_files: Optional[set] = None,
    ):
        super().__init__()
        # self.accept = accept
        self.current_token = current_token
        self.state = state
        self.service = service
        self.api_name = api_name
        self.params = params or {}
        self.current_param = current_param
        self.outfile = outfile
        self.typestate = typestate
        self.efsm = efsm
        self.internal_vars = internal_vars or {}
        self.parsed_calls = parsed_calls or []
        # Track opened files globally
        self.opened_files = opened_files or set()

    def parse_char(self, char: str) -> List["CLIParsingState"]:
        # Handle semicolon as API call separator
        print(f"\n~~~~~~processing char: {char}~~~~~~\n")
        if char == ";": # Only finalize call on semicolon
            # If there's an unprocessed token, process it first
            state = self
            if self.current_token:
                state = self._process_token(self.current_token)
            finalized = state._finalize_call()
            if finalized is None:
                # Ignore empty/invalid call before semicolon
                return [state._reset_for_next_call()]
            call, next_typestate, new_internal_vars, new_opened_files = finalized
            new_calls = state.parsed_calls + [call]
            return [CLIParsingState(
                current_token="",
                state="service",
                service=None,
                api_name=None,
                params={},
                current_param=None,
                outfile=None,
                typestate=next_typestate,
                efsm=self.efsm,
                internal_vars=new_internal_vars,
                parsed_calls=new_calls,
                # Carry forward opened_files
                opened_files=new_opened_files
            )]
        # Tokenize on whitespace
        if char.isspace():
            if self.current_token:
                return [self._process_token(self.current_token)]
            else:
                return [self]
        else:
            # Continue building the current token
            return [CLIParsingState(
                current_token=self.current_token + char,
                state=self.state,
                service=self.service,
                api_name=self.api_name,
                params=self.params.copy(),
                current_param=self.current_param,
                outfile=self.outfile,
                typestate=self.typestate,
                efsm=self.efsm,
                internal_vars=self.internal_vars.copy(),
                parsed_calls=self.parsed_calls.copy(),
                opened_files=self.opened_files.copy()
            )]

    def _process_token(self, token: str) -> "CLIParsingState":
        state = self.state
        service = self.service
        api_name = self.api_name
        params = self.params.copy()
        current_param = self.current_param
        outfile = self.outfile
        typestate = self.typestate
        efsm = self.efsm
        internal_vars = self.internal_vars.copy()
        parsed_calls = self.parsed_calls.copy()
        opened_files = self.opened_files.copy()

        print(f"\n~~~~~~processing token: {token}~~~~~~\n")

        if state == "service":
            service = token
            state = "api"
        elif state == "api":
            api_name = token
            state = "param_or_outfile"
        elif state == "param_or_outfile":
            if token.startswith("--"): 
                current_param = token
                state = "param_value"
            else:
                outfile = token
        elif state == "param_value":
            if current_param is not None:
                params[current_param] = token
            current_param = None
            state = "param_or_outfile"

        return CLIParsingState(
            current_token="",
            state=state,
            service=service,
            api_name=api_name,
            params=params,
            current_param=current_param,
            outfile=outfile,
            typestate=typestate,
            efsm=efsm,
            internal_vars=internal_vars,
            parsed_calls=parsed_calls,
            opened_files=opened_files
        )

    def _finalize_call(self):
        # Only finalize if we have a service and api_name
        if not self.service or not self.api_name:
            return None
        call = {
            "service": self.service,
            "api_name": self.api_name,
            "params": self.params,
            "outfile": self.outfile
        }
        # Compute next typestate and internal variables if EFSM is present
        next_typestate = self.typestate
        new_internal_vars = self.internal_vars.copy()
        if self.efsm:
            symbol = self.api_name
            current_state = self.typestate if self.typestate is not None else self.efsm.initial_state
            transition_result = self.efsm.transition(current_state, symbol, self.params, self.internal_vars)
            if transition_result:
                next_typestate, new_internal_vars = transition_result
            else:
                # No valid transition found - this is an invalid sequence
                print(f"INVALID TRANSITION: {current_state} --{symbol}--> ? (params: {self.params}, vars: {self.internal_vars})")
                return None  # Reject this call
        # Update opened_files if this is an open-file call
        opened_files = self.opened_files.copy()
        if self.api_name == "open-file":
            file_name = self.params.get("--file-name")
            if file_name:
                opened_files.add(file_name)
        return call, next_typestate, new_internal_vars, opened_files

    def _reset_for_next_call(self):
        return CLIParsingState(
            current_token="",
            state="service",
            service=None,
            api_name=None,
            params={},
            current_param=None,
            outfile=None,
            accept=False,
            typestate=self.typestate,
            efsm=self.efsm,
            internal_vars=self.internal_vars.copy(),
            parsed_calls=self.parsed_calls.copy(),
            opened_files=self.opened_files.copy()
        )

    def num_active_states(self):
        return 1

    def finalize(self):
        # Finalize the last call if needed
        state = self
        if self.current_token:
            state = self._process_token(self.current_token)
        finalized = state._finalize_call()
        calls = self.parsed_calls.copy()
        typestate = self.typestate
        internal_vars = self.internal_vars.copy()
        opened_files = self.opened_files.copy()
        if finalized is not None:
            call, typestate, internal_vars, opened_files = finalized
            calls.append(call)
        return {
            "calls": calls,
            "final_typestate": typestate,
            "final_internal_vars": internal_vars,
            "opened_files": opened_files
        }

# Example EFSM for file operations
def create_file_efsm():
    """Create an EFSM for file operations with internal state tracking"""
    return EFSM(
        states=["q1", "q2"],  # q1: no file open, q2: file open
        symbols=["open-file", "read-file", "write-file", "close-file"],
        initial_state="q1",
        final_states=["q1", "q2"],  # Both states can be final
        internal_vars={"f": None},  # f tracks the currently open file
        transitions=[
            # open-file: q1 -> q2, sets f to the file name
            {
                "from_state": "q1",
                "symbol": "open-file",
                "predicate": {},  # No specific predicate needed
                "to_state": "q2",
                "var_updates": {"f": "--file-name"}  # Set f to the file name parameter
            },
            # read-file: q2 -> q2, requires f to match the file name
            {
                "from_state": "q2",
                "symbol": "read-file",
                "predicate": {"--file-name": "f"},  # File name must match current f
                "to_state": "q2",
                "var_updates": {}  # No change to internal vars
            },
            # write-file: q2 -> q2, requires f to match the file name
            {
                "from_state": "q2",
                "symbol": "write-file",
                "predicate": {"--file-name": "f"},  # File name must match current f
                "to_state": "q2",
                "var_updates": {}  # No change to internal vars
            },
            # close-file: q2 -> q1, requires f to match the file name, resets f
            {
                "from_state": "q2",
                "symbol": "close-file",
                "predicate": {"--file-name": "f"},  # File name must match current f
                "to_state": "q1",
                "var_updates": {"f": None}  # Reset f to None
            }
        ]
    )

def get_allowed_values(efsm: EFSM, current_state: str, symbol: str, current_vars: Dict[str, Any]) -> List[str]:
    """
    Get all allowed parameter values for a given state, symbol, and current internal variables.
    This is useful for constrained decoding.
    
    Args:
        efsm: The EFSM
        current_state: Current state
        symbol: API symbol/name
        current_vars: Current internal variables
        
    Returns:
        List of allowed parameter values (for predicates that reference internal variables)
    """
    allowed_values = []
    
    for transition in efsm.transitions:
        if (transition['from_state'] == current_state and 
            transition['symbol'] == symbol):
            
            predicate = transition.get('predicate', {})
            for param_name, expected_value in predicate.items():
                if isinstance(expected_value, str) and expected_value in current_vars:
                    # This predicate references an internal variable
                    allowed_values.append(current_vars[expected_value])
    
    return allowed_values

def get_valid_transitions(efsm: EFSM, current_state: str, current_vars: Dict[str, Any]) -> List[Dict]:
    """
    Get all valid transitions from the current state given the current internal variables.
    
    Args:
        efsm: The EFSM
        current_state: Current state
        current_vars: Current internal variables
        
    Returns:
        List of valid transitions with their symbols and required parameter values
    """
    valid_transitions = []
    
    for transition in efsm.transitions:
        if transition['from_state'] == current_state:
            # Check if this transition is valid given current internal variables
            predicate = transition.get('predicate', {})
            is_valid = True
            
            for param_name, expected_value in predicate.items():
                if isinstance(expected_value, str) and expected_value in current_vars:
                    # This predicate requires a specific internal variable value
                    # We can't determine validity without knowing the actual parameter value
                    pass
            
            valid_transitions.append({
                'symbol': transition['symbol'],
                'predicate': predicate,
                'to_state': transition['to_state'],
                'var_updates': transition.get('var_updates', {})
            })
    
    return valid_transitions

# Example usage:
if __name__ == "__main__":
    efsm = create_file_efsm()
    
    print("=== EFSM Definition ===")
    print(f"States: {efsm.states}")
    print(f"Symbols: {efsm.symbols}")
    print(f"Initial state: {efsm.initial_state}")
    print(f"Final states: {efsm.final_states}")
    print(f"Internal variables: {efsm.internal_vars}")
    print(f"Transitions: {len(efsm.transitions)}")
    
    print("\n=== Testing Valid Sequences ===")
    examples = [
        "fake-service open-file --file-name my-file.txt; fake-service read-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt",
        "fake-service open-file --file-name my-file.txt; fake-service write-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt",
        "fake-service open-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt",
    ]

    for cmd in examples:
        print(f"\n--- Testing: {cmd} ---")
        state = CLIParsingState(typestate=efsm.initial_state, efsm=efsm, internal_vars=efsm.internal_vars.copy())
        for c in cmd:
            state = state.parse_char(c)[0]
        result = state.finalize()
        print(f"Result: {result}")
    
    print("\n=== Testing Invalid Sequences ===")
    invalid_examples = [
        "fake-service read-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt",  # Invalid: read before open
        "fake-service write-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt",  # Invalid: write before open
        "fake-service open-file --file-name file1.txt; fake-service read-file --file-name file2.txt",  # Invalid: wrong file name
    ]

    for cmd in invalid_examples:
        print(f"\n--- Testing: {cmd} ---")
        state = CLIParsingState(typestate=efsm.initial_state, efsm=efsm, internal_vars=efsm.internal_vars.copy())
        for c in cmd:
            state = state.parse_char(c)[0]
        result = state.finalize()
        print(f"Result: {result}")
    
    print("\n=== Decoding Utilities ===")
    # Show how to use the decoding utilities
    current_state = "q1"
    current_vars = {"f": None}
    
    print(f"Current state: {current_state}, vars: {current_vars}")
    valid_transitions = get_valid_transitions(efsm, current_state, current_vars)
    print(f"Valid transitions: {valid_transitions}")
    
    # After opening a file
    current_state = "q2"
    current_vars = {"f": "my-file.txt"}
    print(f"\nCurrent state: {current_state}, vars: {current_vars}")
    valid_transitions = get_valid_transitions(efsm, current_state, current_vars)
    print(f"Valid transitions: {valid_transitions}")
    
    # Show allowed values for read-file
    allowed_values = get_allowed_values(efsm, "q2", "read-file", current_vars)
    print(f"Allowed file names for read-file: {allowed_values}")
