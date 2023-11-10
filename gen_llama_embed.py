# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import re
import random
import numpy as np
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm
from llama import ModelArgs, Transformer, Tokenizer, FunctionLM
from inference_modes import func_embedding_inference, kamel_embedding_inference, vh_embedding_inference
from funchub.math import *


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int):
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
            world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=1, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer


def main(ckpt_dir: str = "/home/sihao/ToolkenGPT/llama-2-7b",
         tokenizer_path: str = "/home/sihao/ToolkenGPT/llama-2-7b/tokenizer.model",
         dataset="funcqa",
         logits_bias: float = 0):
    # set random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    func_semantic_dict_path = f"data/{dataset}/func_semantic_dict.json"
    func_semantic_dict = json.load(open(func_semantic_dict_path, "r"))

    model, tokenizer = load(ckpt_dir, tokenizer_path, local_rank, world_size)
    model.eval()
    new_func_semantic_dict = {}
    with torch.no_grad():
        for op, op_dict in func_semantic_dict.items():
            description = op_dict["description"]
            syntax = op_dict["syntax"]
            description_syntax = description + " " + syntax

            description_token_ids = torch.tensor(tokenizer.encode(description, bos=False, eos=False)).expand(1, -1).to("cuda")
            syntax_token_ids = torch.tensor(tokenizer.encode(syntax, bos=False, eos=False)).expand(1, -1).to("cuda")
            description_syntax_token_ids = torch.tensor(tokenizer.encode(description_syntax, bos=False, eos=False)).expand(1, -1).to("cuda")

            _, h = model(description_token_ids, 0)
            op_dict["description"] = h[0, -1, :].cpu().tolist()
            # op_dict["description"] = ",".join(map(str, h[0, -1, :].cpu().tolist()))

            _, h = model(syntax_token_ids, 0)
            op_dict["syntax"] = h[0, -1, :].cpu().tolist()
            # op_dict["syntax"] = ",".join(map(str, h[0, -1, :].cpu().tolist()))

            _, h = model(description_syntax_token_ids, 0)
            op_dict["combination"] = h[0, -1, :].cpu().tolist()
            # op_dict["combination"] = ",".join(map(str, h[0, -1, :].cpu().tolist()))

            new_func_semantic_dict[op] = op_dict

    with open(f"data/{dataset}/func_embed_dict_llama2-7b.json", "w") as f:
        json.dump(new_func_semantic_dict, f)


if __name__ == "__main__":
    fire.Fire(main)