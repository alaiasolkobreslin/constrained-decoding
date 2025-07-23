from typing import Dict, Optional, List, Any
from parser_base import IncrementalParsingState
import dataclasses

# Minimal DFA/Automaton for typestate tracking
class Automaton:
    def __init__(self, states: List[str], symbols: List[str], transitions: Dict[str, Dict[str, str]], initial_state: str, final_states: List[str]):
        self.states = states
        self.symbols = symbols
        self.transitions = transitions
        self.initial_state = initial_state
        self.final_states = final_states

    def transition(self, current_state: str, symbol: str) -> Optional[str]:
        return self.transitions.get(current_state, {}).get(symbol, None)

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
        # Typestate tracking
        typestate: Optional[str] = None,
        automaton: Optional[Automaton] = None,
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
        self.automaton = automaton
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
            call, next_typestate, new_opened_files = finalized
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
                automaton=self.automaton,
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
                automaton=self.automaton,
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
        automaton = self.automaton
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
            automaton=automaton,
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
        # Compute next typestate if automaton is present
        next_typestate = self.typestate
        if self.automaton:
            symbol = self.api_name
            current_state = self.typestate if self.typestate is not None else self.automaton.initial_state
            next_typestate = self.automaton.transition(current_state, symbol) or current_state
        # Update opened_files if this is an open-file call
        opened_files = self.opened_files.copy()
        if self.api_name == "open-file":
            file_name = self.params.get("--file-name")
            if file_name:
                opened_files.add(file_name)
        return call, next_typestate, opened_files

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
            automaton=self.automaton,
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
        opened_files = self.opened_files.copy()
        if finalized is not None:
            call, typestate, opened_files = finalized
            calls.append(call)
        return {
            "calls": calls,
            "final_typestate": typestate,
            "opened_files": opened_files
        }

# # Example automaton for file operations
# automaton = Automaton(
#     states=["start", "opened", "closed"],
#     symbols=["open-file", "read-file", "write-file", "close-file"],
#     transitions={
#         "start": {"open-file": "opened"},
#         "opened": {"read-file": "opened", "write-file": "opened", "close-file": "closed"},
#         "closed": {}
#     },
#     initial_state="start",
#     final_states=["closed"]
# )

# # Example usage:
# examples = [
#     "fake-service open-file --file-name my-file.txt; fake-service read-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt",
#     "fake-service open-file --file-name my-file.txt; fake-service write-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt",
#     "fake-service open-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt",
#     "fake-service read-file --file-name my-file.txt; fake-service close-file --file-name my-file.txt"  # Invalid: read before open
# ]

# for cmd in examples:
#     state = CLIParsingState(typestate=automaton.initial_state, automaton=automaton)
#     for c in cmd:
#         state = state.parse_char(c)[0]
#     print(state.finalize())
