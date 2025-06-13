from typing import *
from argparse import ArgumentParser
from time import time

import gpt

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("typestate-decoding")
    parser.add_argument("--prompt", type=str, default="file.open(); file.read(); file.")
    parser.add_argument("--strategy", type=str, default="baseline", choices=["baseline", "typestate"])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    # Parameters
    prompt = args.prompt
    strategy = args.strategy
    max_new_tokens = args.max_new_tokens
    do_sample = args.do_sample 
    top_k = args.top_k
    top_p = args.top_p

    if strategy == "baseline":
        llm = gpt.BaselineGPT2(max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p)
    elif strategy == "typestate":
        pass
    
    print(llm.generate(prompt))