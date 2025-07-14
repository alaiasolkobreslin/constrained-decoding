from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Any, Tuple
from types_api import StructurePType, StringPType, IntegerPType, BooleanPType, EnumPType, NullPType, ArrayPType, MapPType, BlobPType, PType

@dataclass
class CLIFlag:
    name: str
    type: PType  # Use types from types_api.py
    required: bool = False
    positional: bool = False
    enum_values: Optional[List[str]] = None  # For EnumPType

@dataclass
class CLICommandSchema:
    name: str
    flags: List[CLIFlag]
    positionals: List[CLIFlag] = field(default_factory=list)

@dataclass
class CLIParserState:
    schema: CLICommandSchema
    parsed_flags: Dict[str, Any] = field(default_factory=dict)
    parsed_positionals: List[Any] = field(default_factory=list)
    expect_flag: Optional[CLIFlag] = None
    current_token: str = ""
    tokens: List[str] = field(default_factory=list)
    accept: bool = False
    error: Optional[str] = None

    def parse_char(self, char: str) -> List['CLIParserState']:
        # Tokenize on whitespace
        if char.isspace():
            if self.current_token:
                return self._process_token(self.current_token)
            else:
                return [self]
        else:
            return [replace(self, current_token=self.current_token + char)]

    def _process_token(self, token: str) -> List['CLIParserState']:
        # If expecting a value for a flag
        if self.expect_flag:
            # Validate type here
            if not self._validate_type(token, self.expect_flag.type, self.expect_flag.enum_values):
                return [replace(self, error=f"Invalid value '{token}' for flag --{self.expect_flag.name}", current_token="")]
            new_flags = self.parsed_flags.copy()
            new_flags[self.expect_flag.name] = token
            return [replace(self, parsed_flags=new_flags, expect_flag=None, current_token="")]
        # If token is a flag
        elif token.startswith("--"):
            flag_name = token[2:]
            flag = next((f for f in self.schema.flags if f.name == flag_name), None)
            if flag:
                return [replace(self, expect_flag=flag, current_token="")]
            else:
                return [replace(self, error=f"Unknown flag --{flag_name}", current_token="")]
        # If token is a positional argument
        else:
            if len(self.parsed_positionals) < len(self.schema.positionals):
                pos_flag = self.schema.positionals[len(self.parsed_positionals)]
                if not self._validate_type(token, pos_flag.type, pos_flag.enum_values):
                    return [replace(self, error=f"Invalid value '{token}' for positional argument {pos_flag.name}", current_token="")]
                new_positionals = self.parsed_positionals + [token]
                return [replace(self, parsed_positionals=new_positionals, current_token="")]
            else:
                return [replace(self, error=f"Unexpected positional argument '{token}'", current_token="")]

    def _validate_type(self, value: str, typ: PType, enum_values: Optional[List[str]]) -> bool:
        # StringPType: always valid
        if isinstance(typ, StringPType):
            return True
        # IntegerPType: must be int
        if isinstance(typ, IntegerPType):
            try:
                int(value)
                return True
            except ValueError:
                return False
        # BooleanPType: must be true/false
        if isinstance(typ, BooleanPType):
            return value.lower() in ("true", "false")
        # EnumPType: must be in allowed values
        if isinstance(typ, EnumPType):
            return value in (enum_values or typ.values)
        # NullPType: must be 'null'
        if isinstance(typ, NullPType):
            return value == "null"
        # BlobPType: accept any string (could add base64 check)
        if isinstance(typ, BlobPType):
            return True
        # Array/Map: not supported for CLI flags
        return True

    def finalize(self) -> Tuple[bool, Optional[str]]:
        # Check required flags/positionals
        for flag in self.schema.flags:
            if flag.required and flag.name not in self.parsed_flags:
                return False, f"Missing required flag --{flag.name}"
        for i, pos in enumerate(self.schema.positionals):
            if pos.required and i >= len(self.parsed_positionals):
                return False, f"Missing required positional argument {pos.name}"
        if self.error:
            return False, self.error
        return True, None

# Example schema for aws s3api get-object
get_object_schema = CLICommandSchema(
    name="get-object",
    flags=[
        CLIFlag("bucket", StringPType(), required=True),
        CLIFlag("key", StringPType(), required=True),
        CLIFlag("version-id", StringPType(), required=False),
        CLIFlag("request-payer", EnumPType(["requester", "owner"]), required=False, enum_values=["requester", "owner"]),
    ],
    positionals=[CLIFlag("outfile", StringPType(), required=True, positional=True)]
)

# Example usage:
if __name__ == "__main__":
    command = "aws s3api get-object --bucket my-bucket --key my-key outfile.txt"
    tokens = command.split()
    # Find the index of the actual command
    try:
        cmd_idx = tokens.index(get_object_schema.name)
    except ValueError:
        print(f"Error: Command '{get_object_schema.name}' not found in input")
        exit(1)
    # Only parse from the command name onward
    to_parse = " ".join(tokens[cmd_idx:])
    state = CLIParserState(schema=get_object_schema)
    for char in to_parse:
        state = state.parse_char(char)[0]
        if state.error:
            print("Error:", state.error)
            break
    else:
        ok, err = state.finalize()
        if ok:
            print("Parsed:", state.parsed_flags, state.parsed_positionals)
        else:
            print("Error:", err) 