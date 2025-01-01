import argparse
import os

import pandas as pd
import polars as pl
from auto_round import AutoRound
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def main(conf: argparse.Namespace) -> None:
    set_seed(1234)

    model_path = conf.model_path
    quant_path = os.path.dirname(model_path) + "/merged_model_quant"

    # load training dataset
    train_long = pl.read_parquet("train_long.parquet")
    fold = 0
    train_dataset = Dataset.from_polars(train_long.filter(train_long["fold"] != fold))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # prepare calibration dataset
    calib_prompts = train_dataset["whole_prompt"]
    tokens = []
    for d in calib_prompts:
        token = tokenizer(d, truncation=True,  return_tensors="pt").data
        tokens.append(token)

    # Quantization
    bits, group_size, sym = 4, 128, True
    autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, dataset=calib_prompts, seqlen=256,
                        nsamples=512,
                        iters=500,
                        )

    autoround.quantize()

    autoround.save_quantized(quant_path, format="auto_gptq", inplace=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the merged model")
    parser.add_argument("--format",
                            help="The format in which to save the model.",
                            default="auto_awq",
                            choices=["auto_gptq", "auto_awq"],
                        )

    args = parser.parse_args()
    main(args)
