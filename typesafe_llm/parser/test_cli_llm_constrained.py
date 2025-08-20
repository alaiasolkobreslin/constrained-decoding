import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from parser_api import CLIParsingState, EFSM, get_valid_transitions, get_allowed_values
from trie import Trie
import json
import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
automaton_json_path = os.path.join(script_dir, "typestates.json")
parameter_names_json_path = os.path.join(script_dir, "parameter_names.json")

with open(automaton_json_path, "r") as f:
	automaton_data = json.load(f)

with open(parameter_names_json_path, "r") as f:
	parameter_names = json.load(f)["fake-service"]

# Setup EFSM and parser
efsm = EFSM(
	states=automaton_data["states"],
	symbols=automaton_data["symbols"],
	initial_state=automaton_data["initial_state"],
	final_states=automaton_data["final_states"],
	internal_vars=automaton_data.get("internal_vars", {}),
	transitions=automaton_data["transitions"]
)

# Load model and tokenizer
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class TrieLogitsProcessor(LogitsProcessor):
	"""Constrain next-token choices to prefixes present in a Trie."""
	def __init__(self, trie, current_prefix_ids, tokenizer):
		self.trie = trie
		self.current_prefix_ids = current_prefix_ids
		self.tokenizer = tokenizer

	def __call__(self, scores):
		allowed = torch.zeros(scores.shape[-1], dtype=torch.bool)
		for token_id in range(scores.shape[-1]):
			next_prefix = self.current_prefix_ids + [token_id]
			node = self.trie
			valid = True
			for tid in next_prefix:
				if tid in node._children:
					node = node._children[tid]
				else:
					valid = False
					break
			if valid:
				allowed[token_id] = True
		# Optional: log a few tokens for debugging
		allowed_tokens = [self.tokenizer.decode([i]) for i in range(scores.shape[-1]) if allowed[i]]
		logger.info(f"[TrieLogitsProcessor] Allowed tokens: {allowed_tokens[:10]}")
		mask = torch.full_like(scores, float('-inf'))
		mask[0, allowed] = 0
		return scores + mask

# =====================
# Constraint abstraction
# =====================
class ConstraintProvider:
	"""
	Interface for language-specific constraint providers.
	Each provider maintains its own incremental parsing state.
	"""
	def reset(self, initial_text: str):
		raise NotImplementedError

	def allowed_lexemes(self) -> list:
		"""Return a list of string lexemes allowed at the current position. Empty/None = unconstrained."""
		raise NotImplementedError

	def separators(self) -> list:
		"""Return the list of separator lexemes to emit between lexemes when required (e.g., " ", ";", "\n")."""
		raise NotImplementedError

	def on_token_emitted(self, token_str: str):
		"""Advance internal parsing state using the raw decoded token string (may be multiple chars)."""
		raise NotImplementedError

class CLIConstraintProvider(ConstraintProvider):
	"""EFSM-backed provider for the CLI language used in this repo."""
	# Mapping from parameter names to state attribute names for value sets
	param_value_sources = {
		"--file-name": "opened_files",
	}

	def __init__(self, efsm: EFSM, parameter_names_dict: dict):
		self.efsm = efsm
		self.parameter_names_dict = parameter_names_dict
		self.state = None
		self.current_api_name = None
		self.current_param_name = None

	def reset(self, initial_text: str):
		self.state = CLIParsingState(typestate=self.efsm.initial_state, efsm=self.efsm, internal_vars=self.efsm.internal_vars.copy())
		self.current_api_name = None
		self.current_param_name = None
		for ch in initial_text:
			self.state = self.state.parse_char(ch)[0]

	def _unused_params(self) -> list:
		all_params = self.parameter_names_dict.get(self.current_api_name, []) if self.current_api_name is not None else []
		used_params = set(self.state.params.keys())
		return [p for p in all_params if p not in used_params]

	def allowed_lexemes(self) -> list:
		if self.state.state == "service":
			return ["fake-service"]
		if self.state.state == "api" or self.current_api_name is None:
			valid_transitions = get_valid_transitions(self.efsm, self.state.typestate, self.state.internal_vars)
			return [t['symbol'] for t in valid_transitions]
		if self.state.state == "param_or_outfile":
			if self.current_param_name is None:
				unused = self._unused_params()
				return unused
			else:
				# expecting a value next
				return self._allowed_values_for_current_param()
		if self.state.state == "param_value":
			return self._allowed_values_for_current_param()
		return []

	def _allowed_values_for_current_param(self) -> list:
		if self.current_api_name is None or self.current_param_name is None:
			return []
		allowed = get_allowed_values(self.efsm, self.state.typestate, self.current_api_name, self.state.internal_vars)
		if allowed:
			return allowed
		# Fallback: pluggable value sets from parsing state
		value_set = None
		if self.current_param_name in self.param_value_sources:
			value_set_name = self.param_value_sources[self.current_param_name]
			value_set = getattr(self.state, value_set_name, None)
		if value_set is not None:
			return list(value_set)
		return []

	def separators(self) -> list:
		# Prefer semicolon when no more params are available at param_or_outfile.
		# If no more params remain for the current API call, prefer semicolon to finalize
		if self.current_api_name is not None and len(self._unused_params()) == 0:
			return [";"]
		# Default: a single space separates lexemes
		return [" "]

	def on_token_emitted(self, token_str: str):
		# Update higher-level tracking based on incremental parsing transitions
		for ch in token_str:
			prev_state = self.state
			self.state = self.state.parse_char(ch)[0]
			# When a space finalizes a token, inspect the NEW state to capture completed units
			if ch.isspace():
				# API name just finalized: moved from api -> param_or_outfile
				if prev_state.state == "api" and self.state.state == "param_or_outfile" and self.state.api_name:
					self.current_api_name = self.state.api_name
				# Param name just finalized: moved from param_or_outfile -> param_value
				if prev_state.state == "param_or_outfile" and self.state.state == "param_value" and self.state.current_param:
					self.current_param_name = self.state.current_param
				# Param value just finalized: moved from param_value -> param_or_outfile
				if prev_state.state == "param_value" and self.state.state == "param_or_outfile":
					self.current_param_name = None
			# On semicolon, reset for the next call
			if ch == ";":
				self.current_api_name = None
				self.current_param_name = None

class PythonConstraintProvider(ConstraintProvider):
	"""
	Minimal, stubbed provider for a tiny subset of Python to demonstrate the abstraction.
	Grammar (toy):
		toplevel -> (def func | return literal) NEWLINE
		func -> "def" WS name "(" ")" ":" NEWLINE
		name in {foo, bar}
		literal in {0, 1}
	"""
	def __init__(self, allowed_function_names=None):
		self.allowed_function_names = allowed_function_names or ["foo", "bar"]
		self.mode = "toplevel"
		self.just_completed_lexeme = None

	def reset(self, initial_text: str):
		self.mode = "toplevel"
		self.just_completed_lexeme = None
		# Stateless scan: switch modes based on simple substrings
		for ch in initial_text:
			self.on_token_emitted(ch)

	def allowed_lexemes(self) -> list:
		if self.mode == "toplevel":
			return ["def", "return"]
		if self.mode == "after_def":
			return self.allowed_function_names
		if self.mode == "after_return":
			return ["0", "1"]
		return []

	def separators(self) -> list:
		if self.mode in ("toplevel", "after_def", "after_return", "after_name"):
			return [" ", "(", ")", ":", "\n"]
		return [" "]

	def on_token_emitted(self, token_str: str):
		for ch in token_str:
			# Crude mode transitions
			if self.mode == "toplevel":
				# Detect if we just typed 'def' or 'return' followed by space
				pass
			if ch == " ":
				if self.just_completed_lexeme == "def":
					self.mode = "after_def"
				elif self.just_completed_lexeme == "return":
					self.mode = "after_return"
				self.just_completed_lexeme = None
			elif ch == "\n":
				self.mode = "toplevel"
				self.just_completed_lexeme = None
			elif ch == ":":
				self.mode = "toplevel"
			elif ch.isalpha():
				# When we finish a name, we'll set just_completed_lexeme externally
				pass

	def note_completed_lexeme(self, lexeme: str):
		# Helper to be called by the generator when a full lexeme is matched
		self.just_completed_lexeme = lexeme

class ConstrainedGenerator:
	"""
	Language-agnostic constrained generator that delegates allowed lexemes/separators
	to a ConstraintProvider.
	"""
	def __init__(self, tokenizer, model, provider: ConstraintProvider, max_steps=40):
		self.tokenizer = tokenizer
		self.model = model
		self.provider = provider
		self.max_steps = max_steps

	def build_trie(self, options):
		trie = Trie()
		for opt in options:
			ids = self.tokenizer.encode(opt, add_special_tokens=False)
			trie.insert(ids, opt)
		return trie

	def mask_with_trie(self, trie, prefix_ids, logits):
		trie_logits_processor = TrieLogitsProcessor(trie, prefix_ids, self.tokenizer)
		return trie_logits_processor(logits)

	def _decode_and_emit(self, input_ids, next_token_id):
		token_str = self.tokenizer.decode(next_token_id)
		logger.info(f"[ConstrainedGenerator] Emitting token: '{token_str}'")
		self.provider.on_token_emitted(token_str)
		return torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1), token_str

	def run(self, prompt: str):
		input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
		self.provider.reset(prompt)
		current_prefix_ids = []
		expecting_separator = False
		last_allowed_lexemes = []
		for step in range(self.max_steps):
			outputs = self.model(input_ids)
			logits = outputs.logits[:, -1, :]
			if expecting_separator:
				seps = self.provider.separators()
				logger.info(f"[run] expecting separator; seps={seps}")
				if not seps:
					# fallback: allow space
					seps = [" "]
				sep_trie = self.build_trie(seps)
				filtered = self.mask_with_trie(sep_trie, [], logits)
				next_token_id = torch.argmax(filtered, dim=-1)
				input_ids, token_str = self._decode_and_emit(input_ids, next_token_id)
				# Consider any emitted separator as completion of separator phase
				expecting_separator = False
				current_prefix_ids = []
				continue
			allowed = self.provider.allowed_lexemes()
			last_allowed_lexemes = allowed or []
			if not allowed:
				# Fall back to structural separators provided by the provider
				seps = self.provider.separators()
				logger.info(f"[run] no lexemes; using separators seps={seps}")
				if not seps:
					seps = [" "]
				sep_trie = self.build_trie(seps)
				filtered = self.mask_with_trie(sep_trie, [], logits)
				next_token_id = torch.argmax(filtered, dim=-1)
				input_ids, token_str = self._decode_and_emit(input_ids, next_token_id)
				current_prefix_ids = []
				# After emitting a separator, we may still be in separator phase depending on provider
				expecting_separator = False
				continue
			lexeme_trie = self.build_trie(allowed)
			filtered = self.mask_with_trie(lexeme_trie, current_prefix_ids, logits)
			next_token_id = torch.argmax(filtered, dim=-1)
			current_prefix_ids.append(next_token_id.item())
			# Check for completed lexeme
			completed = None
			allowed_token_ids = [self.tokenizer.encode(lx, add_special_tokens=False) for lx in allowed]
			for lx_ids, lx in zip(allowed_token_ids, allowed):
				if current_prefix_ids == lx_ids:
					completed = lx
					break
			input_ids, token_str = self._decode_and_emit(input_ids, next_token_id)
			if completed is not None:
				logger.info(f"[run] Completed lexeme: '{completed}'")
				# If provider exposes a hook, notify it
				if hasattr(self.provider, "note_completed_lexeme"):
					try:
						self.provider.note_completed_lexeme(completed)
					except Exception:
						pass
				current_prefix_ids = []
				expecting_separator = True
			# Optional stop: if model starts outputting too many newlines on Python
			if token_str == ";":
				# For CLI, semicolon can indicate end of call; keep going if more steps remain
				pass
		logger.info("\nFinal output: " + self.tokenizer.decode(input_ids[0]))

if __name__ == "__main__":
	# CLI demo (preserves previous behavior)
	prompt = "fake-service open-file --file-name my-file.txt; fake-service "
	cli_provider = CLIConstraintProvider(efsm, parameter_names)
	generator = ConstrainedGenerator(tokenizer, model, cli_provider, max_steps=40)
	generator.run(prompt)
