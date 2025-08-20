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

class CLIConstrainedGenerator:
	"""
	Constrained decoding for a simple CLI: service -> api -> [params|outfile].
	Uses EFSM to restrict valid APIs and predicates to restrict param values.
	"""
	# Mapping from parameter names to state attribute names for value sets
	param_value_sources = {
		"--file-name": "opened_files",
	}

	def __init__(self, efsm, parameter_names_dict, tokenizer, model, max_steps=15):
		self.efsm = efsm
		self.parameter_names_dict = parameter_names_dict
		self.tokenizer = tokenizer
		self.model = model
		self.max_steps = max_steps

	def build_trie(self, options):
		"""Build a Trie from a list of string options using the tokenizer."""
		trie = Trie()
		for opt in options:
			ids = self.tokenizer.encode(opt, add_special_tokens=False)
			trie.insert(ids, opt)
		return trie

	def mask_with_trie(self, trie, prefix_ids, logits):
		trie_logits_processor = TrieLogitsProcessor(trie, prefix_ids, self.tokenizer)
		return trie_logits_processor(logits)

	def mask_separator(self, logits):
		"""Allow only space as a separator token."""
		allowed = torch.zeros(logits.shape[-1], dtype=torch.bool)
		for separator in [" "]:
			allowed[self.tokenizer.encode(separator, add_special_tokens=False)] = True
		mask = torch.full_like(logits, float('-inf'))
		mask[0, allowed] = 0
		return logits + mask

	def mask_cli_structure(self, state, logits):
		"""Restrict tokens based on parser state."""
		allowed = torch.zeros(logits.shape[-1], dtype=torch.bool)
		if state.state == "param_or_outfile":
			for separator in ["--", ";"]:
				allowed[self.tokenizer.encode(separator, add_special_tokens=False)] = True
		else:
			allowed[:] = True
		mask = torch.full_like(logits, float('-inf'))
		mask[0, allowed] = 0
		return logits + mask

	def handle_separator(self, logits):
		"""Emit a space or a semicolon when a separator is expected."""
		filtered_logits = self.mask_separator(logits)
		next_token_id = torch.argmax(filtered_logits, dim=-1)
		next_token_str = self.tokenizer.decode(next_token_id)
		if next_token_str == " " or next_token_str == ";":
			return next_token_id, next_token_str, False
		else:
			raise ValueError(f"Expected whitespace or semicolon, got {next_token_str}")

	def handle_api_name(self, state, api_name_prefix, logits):
		"""Decode API name using EFSM valid transitions and a Trie mask."""
		valid_transitions = get_valid_transitions(self.efsm, state.typestate, state.internal_vars)
		valid_api_names = [t['symbol'] for t in valid_transitions]
		logger.info(f"[handle_api_name] Current state: {state.typestate}, internal_vars: {state.internal_vars}")
		logger.info(f"[handle_api_name] Valid API names: {valid_api_names}")
		valid_api_name_trie = self.build_trie(valid_api_names)
		valid_api_name_token_ids = [self.tokenizer.encode(api, add_special_tokens=False) for api in valid_api_names]
		filtered_logits = self.mask_with_trie(valid_api_name_trie, api_name_prefix, logits)
		next_token_id = torch.argmax(filtered_logits, dim=-1)
		api_name_prefix.append(next_token_id.item())
		current_api_name = None
		expecting_separator = False
		for api_ids, api in zip(valid_api_name_token_ids, valid_api_names):
			if api_name_prefix == api_ids:
				logger.info(f"Completed API name: {self.tokenizer.decode(api_name_prefix)}")
				current_api_name = api
				api_name_prefix.clear()
				expecting_separator = True
				break
		next_token_str = self.tokenizer.decode(next_token_id)
		return next_token_id, next_token_str, current_api_name, expecting_separator

	def handle_param_name(self, current_api_name, param_name_prefix, logits, state):
		"""Decode parameter name for the current API; if none left, force ';'."""
		all_params = self.parameter_names_dict.get(current_api_name, []) if current_api_name is not None else []
		used_params = set(state.params.keys())
		unused_params = [p for p in all_params if p not in used_params]
		logger.info(f"[handle_param_name] current_api_name: {current_api_name}, all_params: {all_params}")
		logger.info(f"[handle_param_name] used_params: {used_params}, unused_params: {unused_params}")
		if unused_params:
			param_name_trie = self.build_trie(unused_params)
			filtered_logits = self.mask_with_trie(param_name_trie, param_name_prefix, logits)
		else:
			logger.info("All parameters used, only allowing semicolon separator")
			semicolon_token_id = self.tokenizer.encode(";", add_special_tokens=False)[0]
			filtered_logits = torch.full_like(logits, float('-inf'))
			filtered_logits[0, semicolon_token_id] = logits[0, semicolon_token_id]
		next_token_id = torch.argmax(filtered_logits, dim=-1)
		param_name_prefix.append(next_token_id.item())
		expecting_separator = False
		completed_param_name = None
		if unused_params:
			param_name_token_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in unused_params]
			for param_ids, param_name in zip(param_name_token_ids, unused_params):
				if param_name_prefix == param_ids:
					logger.info(f"Completed parameter name: {self.tokenizer.decode(param_name_prefix)}")
					completed_param_name = param_name
					param_name_prefix.clear()
					expecting_separator = True
					break
			else:
				logger.info(f"Didn't complete parameter name! Prefix is: {self.tokenizer.decode(param_name_prefix)}")
		else:
			if next_token_id.item() == self.tokenizer.encode(";", add_special_tokens=False)[0]:
				logger.info("Generated semicolon separator")
				param_name_prefix.clear()
				expecting_separator = True
			else:
				logger.info(f"Unexpected token when expecting semicolon: {self.tokenizer.decode(next_token_id)}")
		next_token_str = self.tokenizer.decode(next_token_id)
		return next_token_id, next_token_str, expecting_separator, completed_param_name

	def handle_param_value(self, current_api_name, param_name, param_value_prefix, logits, state):
		"""Decode parameter value; if EFSM provides allowed values, constrain to those."""
		allowed_values = get_allowed_values(self.efsm, state.typestate, current_api_name, state.internal_vars)
		logger.info(f"[handle_param_value] param_name: {param_name}, allowed_values: {allowed_values}")
		if allowed_values:
			value_trie = self.build_trie(allowed_values)
			filtered_logits = self.mask_with_trie(value_trie, param_value_prefix, logits)
		else:
			value_set = None
			if param_name in self.param_value_sources:
				value_set_name = self.param_value_sources[param_name]
				value_set = getattr(state, value_set_name, None)
			if value_set is not None:
				allowed_values = list(value_set)
				value_trie = self.build_trie(allowed_values)
				filtered_logits = self.mask_with_trie(value_trie, param_value_prefix, logits)
			else:
				filtered_logits = logits
		next_token_id = torch.argmax(filtered_logits, dim=-1)
		param_value_prefix.append(next_token_id.item())
		expecting_separator = False
		if allowed_values:
			value_token_ids = [self.tokenizer.encode(v, add_special_tokens=False) for v in allowed_values]
			for value_ids in value_token_ids:
				if param_value_prefix == value_ids:
					logger.info(f"Completed value: {self.tokenizer.decode(param_value_prefix)}")
					param_value_prefix.clear()
					expecting_separator = True
					break
		else:
			token_str = self.tokenizer.decode(next_token_id)
			if token_str.isspace() or token_str == ";":
				param_value_prefix.clear()
				expecting_separator = True
		next_token_str = self.tokenizer.decode(next_token_id)
		return next_token_id, next_token_str, expecting_separator

	def handle_cli_structure(self, state, logits):
		"""Handle structure-only masking (e.g., in param_or_outfile)."""
		filtered_logits = self.mask_cli_structure(state, logits)
		next_token_id = torch.argmax(filtered_logits, dim=-1)
		next_token_str = self.tokenizer.decode(next_token_id)
		return next_token_id, next_token_str

	def run(self, prompt):
		input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
		state = CLIParsingState(typestate=self.efsm.initial_state, efsm=self.efsm, internal_vars=self.efsm.internal_vars.copy())
		for char in prompt:
			state = state.parse_char(char)[0]
		api_name_prefix = []
		param_name_prefix = []
		param_value_prefix = []
		constraining_api_name = False
		expecting_separator = False
		current_api_name = None
		current_param_name = None
		service_completed = False
		for step in range(self.max_steps):
			logger.info(f"\n[run] Step {step}: parser_state.state = {state.state}, current_api_name = {current_api_name}, current_param_name = {current_param_name}, decoded_so_far = '{self.tokenizer.decode(input_ids[0])}'")
			logger.info(f"[run] Step {step}: opened_files = {state.opened_files}, internal_vars = {state.internal_vars}")
			outputs = self.model(input_ids)
			logits = outputs.logits[:, -1, :]
			if expecting_separator:
				next_token_id, next_token_str, expecting_separator = self.handle_separator(logits)
			elif state.state == "api" or constraining_api_name or service_completed:
				# Continue generating/finishing API name
				next_token_id, next_token_str, new_api_name, new_expecting_separator = self.handle_api_name(state, api_name_prefix, logits)
				if new_api_name is not None:
					current_api_name = new_api_name
					constraining_api_name = False
				expecting_separator = new_expecting_separator
				if service_completed:
					service_completed = False
					constraining_api_name = True
			elif state.state == "param_or_outfile":
				if current_param_name is None:
					logger.info(f"[run] About to generate parameter name for API: {current_api_name}")
					next_token_id, next_token_str, expecting_separator, completed_param_name = self.handle_param_name(current_api_name, param_name_prefix, logits, state)
					if expecting_separator and completed_param_name is not None:
						current_param_name = completed_param_name
				else:
					next_token_id, next_token_str, expecting_separator = self.handle_param_value(current_api_name, current_param_name, param_value_prefix, logits, state)
					if expecting_separator:
						current_param_name = None
			elif state.state == "param_value":
				next_token_id, next_token_str, expecting_separator = self.handle_param_value(current_api_name, current_param_name, param_value_prefix, logits, state)
				if expecting_separator:
					current_param_name = None
			elif state.state == "service":
				logger.info("[run] Generating service name")
				if not hasattr(self, 'service_prefix'):
					self.service_prefix = []
				service_trie = self.build_trie(["fake-service"])
				filtered_logits = self.mask_with_trie(service_trie, self.service_prefix, logits)
				next_token_id = torch.argmax(filtered_logits, dim=-1)
				next_token_str = self.tokenizer.decode(next_token_id)
				self.service_prefix.append(next_token_id.item())
				service_token_ids = self.tokenizer.encode("fake-service", add_special_tokens=False)
				if self.service_prefix == service_token_ids:
					logger.info("Completed service name: fake-service")
					self.service_prefix.clear()
					service_completed = True
					expecting_separator = True  # ensure a space follows service
				else:
					logger.info(f"Service name prefix: {self.tokenizer.decode(self.service_prefix)}")
			else:
				next_token_id, next_token_str = self.handle_cli_structure(state, logits)
			logger.info(f"[run] Step {step}: generated token '{next_token_str}'")
			input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
			state = state.parse_char(next_token_str)[0]
			if next_token_str == ";":
				logger.info("[run] Semicolon processed, state and opened_files updated for next call.")
			if state.typestate in self.efsm.final_states:
				logger.info("Reached final typestate!")
				break
		logger.info("\nFinal output: " + self.tokenizer.decode(input_ids[0]))
		logger.info("Parser state: " + str(state.finalize()))

if __name__ == "__main__":
	prompt = "fake-service open-file --file-name my-file.txt; fake-service "
	generator = CLIConstrainedGenerator(efsm, parameter_names, tokenizer, model, max_steps=40)
	generator.run(prompt)
