from typing import Dict, Optional

def parse_cli_command(command: str):
    tokens = command.strip().split()
    if len(tokens) < 2:
        raise ValueError("Command must have at least service and api-name")
    service = tokens[0]
    api_name = tokens[1]
    params: Dict[str, str] = {}
    outfile: Optional[str] = None

    i = 2
    while i < len(tokens):
        if tokens[i].startswith("--"):
            param_name = tokens[i][2:]
            if i + 1 < len(tokens) and not tokens[i+1].startswith("--"):
                param_value = tokens[i+1]
                i += 2
            else:
                param_value = ""  # or None, if you want to allow flags with no value
                i += 1
            params[param_name] = param_value
        else:
            # If it's the last token, treat as outfile
            if i == len(tokens) - 1:
                outfile = tokens[i]
                i += 1
            else:
                # Unexpected positional, but just skip or collect as needed
                i += 1

    return {
        "service": service,
        "api_name": api_name,
        "params": params,
        "outfile": outfile
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
    print(parse_cli_command(cmd))
