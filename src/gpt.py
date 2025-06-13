from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

from process_logits import TypeStateConstrainedLogitsProcessor

class LLM:

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def eval(self):
        self.model.eval()

    def generate(self, prompt):
        pass


class BaselineGPT2(LLM):

    def __init__(self, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        super().__init__(tokenizer, model)
        self.eval()

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p
            )

        # Decode and return generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

class TypestateConstrainedGPT2(LLM):

    def init(self, typestate, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.logits_processor = TypeStateConstrainedLogitsProcessor(typestate)
        super().__init__(tokenizer, model)
        self.eval()

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.logits_processor.process(inputs)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
