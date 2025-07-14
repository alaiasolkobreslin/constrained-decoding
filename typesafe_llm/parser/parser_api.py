from typing import Dict, Optional

class IncrementalCLIParser:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tokens = []
        self.current_token = ""
        self.state = "service"  # or "api", "param", "value", "outfile", etc.
        self.service = None
        self.api_name = None
        self.params: Dict[str, str] = {}
        self.current_param = None
        self.outfile = None

    def parse_char(self, char: str):
        if char.isspace():
            if self.current_token:
                self._process_token(self.current_token)
                self.current_token = ""
        else:
            self.current_token += char

    def _process_token(self, token: str):
        if self.state == "service":
            self.service = token
            self.state = "api"
        elif self.state == "api":
            self.api_name = token
            self.state = "param_or_outfile"
        elif self.state == "param_or_outfile":
            if token.startswith("--"):
                self.current_param = token[2:]
                self.state = "param_value"
            else:
                # If this is the last token, treat as outfile
                self.outfile = token
        elif self.state == "param_value":
            self.params[self.current_param] = token
            self.current_param = None
            self.state = "param_or_outfile"

    def finalize(self):
        # If there's a token left, process it
        if self.current_token:
            self._process_token(self.current_token)
            self.current_token = ""
        return {
            "service": self.service,
            "api_name": self.api_name,
            "params": self.params,
            "outfile": self.outfile
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
    parser = IncrementalCLIParser()
    for c in cmd:
        parser.parse_char(c)
    print(parser.finalize())
