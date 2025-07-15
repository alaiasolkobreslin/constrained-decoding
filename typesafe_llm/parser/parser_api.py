from typing import Dict, Optional, List
from parser_base import IncrementalParsingState

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
    ):
        super().__init__(accept=accept)
        self.current_token = current_token
        self.state = state
        self.service = service
        self.api_name = api_name
        self.params = params or {}
        self.current_param = current_param
        self.outfile = outfile

    def parse_char(self, char: str) -> List["CLIParsingState"]:
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
                accept=False
            )]

    def _process_token(self, token: str) -> "CLIParsingState":
        state = self.state
        service = self.service
        api_name = self.api_name
        params = self.params.copy()
        current_param = self.current_param
        outfile = self.outfile
        accept = False

        if state == "service":
            service = token
            state = "api"
        elif state == "api":
            api_name = token
            state = "param_or_outfile"
        elif state == "param_or_outfile":
            if token.startswith("--"):
                current_param = token[2:]
                state = "param_value"
            else:
                outfile = token
                accept = True
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
            accept=accept
        )

    def num_active_states(self):
        return 1

    def finalize(self):
        state = self
        if self.current_token:
            state = self._process_token(self.current_token)
        return {
            "service": state.service,
            "api_name": state.api_name,
            "params": state.params,
            "outfile": state.outfile
        }

# Example usage:
examples = [
    "s3api get-object --bucket my-bucket --key my-key outfile.txt",
    "dynamodb put-item --table-name my-table --item '{\"id\": {\"S\": \"123\"}}'",
    "ec2 run-instances --image-id ami-123456 --instance-type t2.micro --min-count 1 --max-count 2",
    "s3api create-bucket --bucket my-bucket",
    "fake-api open-file --file-name my-file.txt"
]

for cmd in examples:
    state = CLIParsingState()
    for c in cmd:
        state = state.parse_char(c)[0]
    print(state.finalize())
