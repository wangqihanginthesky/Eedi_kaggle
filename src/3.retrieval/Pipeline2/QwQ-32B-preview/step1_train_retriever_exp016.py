# %%
DATA_PATH = "../../../../data/retrieve_train"
MODEL_NAME = "Qwen/QwQ-32B-Preview"
OUTPUT_PATH = "./exp016-qwq"
MODEL_OUTPUT_PATH = f"{OUTPUT_PATH}/output_retrieval_short"

RETRIEVE_NUM = 25
SEED = 9283
EPOCH = 10
LR = 4e-05
BS = 32

TRAINING = True
DEBUG = False
WANDB = False
REPORT_TO = "none"

# %%
import gc
import os
import random

import datasets
import numpy as np
import pandas as pd
import polars as pl
import sentence_transformers
import wandb
from accelerate.utils import release_memory
from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, TripletEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sklearn.metrics.pairwise import cosine_similarity
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from peft import prepare_model_for_kbit_training, PeftModelForFeatureExtraction
from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from llm2vec import LLM2Vec

# %%
import json
import logging
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

import huggingface_hub
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, MT5Config, T5Config
from transformers.utils import is_peft_available
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _save_pretrained_wrapper(_save_pretrained_fn: Callable, subfolder: str) -> Callable[..., None]:
    def wrapper(save_directory: str | Path, **kwargs) -> None:
        os.makedirs(Path(save_directory) / subfolder, exist_ok=True)
        return _save_pretrained_fn(Path(save_directory) / subfolder, **kwargs)

    return wrapper

class CustomTransformer(models.Transformer):
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int | None = None,
        model_args: dict[str, Any] | None = None,
        tokenizer_args: dict[str, Any] | None = None,
        config_args: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str = None,
        backend: str = "torch",
    ) -> None:
        super().__init__(model_name_or_path)
        self.config_keys = ["max_seq_length", "do_lower_case"]
        self.do_lower_case = do_lower_case
        self.backend = backend
        if model_args is None:
            model_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}
        if config_args is None:
            config_args = {}

        config = AutoConfig.from_pretrained(model_name_or_path, **config_args, cache_dir=cache_dir)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.auto_model = LLM2Vec.from_pretrained(
            model_name_or_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            use_cache=False,
        ).model
        config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        self.auto_model = get_peft_model(self.auto_model, config)
        print(self.auto_model.print_trainable_parameters())

        if max_seq_length is not None and "model_max_length" not in tokenizer_args:
            tokenizer_args["model_max_length"] = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args,
        )

        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if (
                hasattr(self.auto_model, "config")
                and hasattr(self.auto_model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def _load_model(self, model_name_or_path, config, cache_dir, backend, **model_args) -> None:
        """Loads the transformer model"""
        if backend == "torch":
            if isinstance(config, T5Config):
                self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
            elif isinstance(config, MT5Config):
                self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
            else:
                self.auto_model = AutoModel.from_pretrained(
                    model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                )
        elif backend == "onnx":
            self._load_onnx_model(model_name_or_path, config, cache_dir, **model_args)
        elif backend == "openvino":
            self._load_openvino_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            raise ValueError(f"Unsupported backend '{backend}'. `backend` should be `torch`, `onnx`, or `openvino`.")

    def _load_openvino_model(self, model_name_or_path, config, cache_dir, **model_args) -> None:
        if isinstance(config, T5Config) or isinstance(config, MT5Config):
            raise ValueError("T5 models are not yet supported by the OpenVINO backend.")

        try:
            from optimum.intel import OVModelForFeatureExtraction
            from optimum.intel.openvino import OV_XML_FILE_NAME
        except ModuleNotFoundError:
            raise Exception(
                "Using the OpenVINO backend requires installing Optimum and OpenVINO. "
                "You can install them with pip: `pip install optimum[openvino]`."
            )

        load_path = Path(model_name_or_path)
        is_local = load_path.exists()
        backend_name = "OpenVINO"
        target_file_glob = "openvino*.xml"

        # Determine whether the model should be exported or whether we can load it directly
        export, model_args = self._backend_should_export(
            load_path, is_local, model_args, OV_XML_FILE_NAME, target_file_glob, backend_name
        )

        # If we're exporting, then there's no need for a file_name to load the model from
        if export:
            model_args.pop("file_name", None)

        # ov_config can be either a dictionary, or point to a json file with an OpenVINO config
        if "ov_config" in model_args:
            ov_config = model_args["ov_config"]
            if not isinstance(ov_config, dict):
                if not Path(ov_config).exists():
                    raise ValueError(
                        "ov_config should be a dictionary or a path to a .json file containing an OpenVINO config"
                    )
                with open(ov_config, encoding="utf-8") as f:
                    model_args["ov_config"] = json.load(f)
        else:
            model_args["ov_config"] = {}

        # Either load an exported model, or export the model to OpenVINO
        self.auto_model: OVModelForFeatureExtraction = OVModelForFeatureExtraction.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            export=export,
            **model_args,
        )
        # Wrap the save_pretrained method to save the model in the correct subfolder
        self.auto_model._save_pretrained = _save_pretrained_wrapper(self.auto_model._save_pretrained, self.backend)

        # Warn the user to save the model if they haven't already
        if export:
            self._backend_warn_to_save(model_name_or_path, is_local, backend_name)

    def _load_onnx_model(self, model_name_or_path, config, cache_dir, **model_args) -> None:
        try:
            import onnxruntime as ort
            from optimum.onnxruntime import ONNX_WEIGHTS_NAME, ORTModelForFeatureExtraction
        except ModuleNotFoundError:
            raise Exception(
                "Using the ONNX backend requires installing Optimum and ONNX Runtime. "
                "You can install them with pip: `pip install optimum[onnxruntime]` "
                "or `pip install optimum[onnxruntime-gpu]`"
            )

        # Default to the highest priority available provider if not specified
        # E.g. Tensorrt > CUDA > CPU
        model_args["provider"] = model_args.pop("provider", ort.get_available_providers()[0])

        load_path = Path(model_name_or_path)
        is_local = load_path.exists()
        backend_name = "ONNX"
        target_file_glob = "*.onnx"

        # Determine whether the model should be exported or whether we can load it directly
        export, model_args = self._backend_should_export(
            load_path, is_local, model_args, ONNX_WEIGHTS_NAME, target_file_glob, backend_name
        )

        # If we're exporting, then there's no need for a file_name to load the model from
        if export:
            model_args.pop("file_name", None)

        # Either load an exported model, or export the model to ONNX
        self.auto_model: ORTModelForFeatureExtraction = ORTModelForFeatureExtraction.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            export=export,
            **model_args,
        )
        # Wrap the save_pretrained method to save the model in the correct subfolder
        self.auto_model._save_pretrained = _save_pretrained_wrapper(self.auto_model._save_pretrained, self.backend)

        # Warn the user to save the model if they haven't already
        if export:
            self._backend_warn_to_save(model_name_or_path, is_local, backend_name)

    def _backend_should_export(
        self,
        load_path: Path,
        is_local: bool,
        model_args: dict[str, Any],
        target_file_name: str,
        target_file_glob: str,
        backend_name: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Determines whether the model should be exported to the backend, or if it can be loaded directly.
        Also update the `file_name` and `subfolder` model_args if necessary.

        These are the cases:

        1. If export is set in model_args, just return export
        2. If `<subfolder>/<file_name>` exists; set export to False
        3. If `<backend>/<file_name>` exists; set export to False and set subfolder to the backend (e.g. "onnx")
        4. If `<file_name>` contains a folder, add those folders to the subfolder and set the file_name to the last part

        We will warn if:

        1. The expected file does not exist in the model directory given the optional file_name and subfolder.
           If there are valid files for this backend, but they're don't align with file_name, then we give a useful warning.
        2. Multiple files are found in the model directory that match the target file name and the user did not
           specify the desired file name via `model_kwargs={"file_name": "<file_name>"}`

        Args:
            load_path: The model repository or directory, as a Path instance
            is_local: Whether the model is local or remote, i.e. whether load_path is a local directory
            model_args: The model_args dictionary. Notable keys are "export", "file_name", and "subfolder"
            target_file_name: The expected file name in the model directory, e.g. "model.onnx" or "openvino_model.xml"
            target_file_glob: The glob pattern to match the target file name, e.g. "*.onnx" or "openvino*.xml"
            backend_name: The human-readable name of the backend for use in warnings, e.g. "ONNX" or "OpenVINO"

        Returns:
            Tuple[bool, dict[str, Any]]: A tuple of the export boolean and the updated model_args dictionary.
        """

        export = model_args.pop("export", None)
        if export is not None:
            return export, model_args

        file_name = model_args.get("file_name", target_file_name)
        subfolder = model_args.get("subfolder", None)
        primary_full_path = Path(subfolder, file_name).as_posix() if subfolder else Path(file_name).as_posix()
        secondary_full_path = (
            Path(subfolder, self.backend, file_name).as_posix()
            if subfolder
            else Path(self.backend, file_name).as_posix()
        )
        glob_pattern = f"{subfolder}/**/{target_file_glob}" if subfolder else f"**/{target_file_glob}"

        # Get the list of files in the model directory that match the target file name
        if is_local:
            model_file_names = [path.relative_to(load_path).as_posix() for path in load_path.glob(glob_pattern)]
        else:
            all_files = huggingface_hub.list_repo_files(
                load_path.as_posix(),
                repo_type="model",
                revision=model_args.get("revision", None),
                token=model_args.get("token", None),
            )
            model_file_names = [fname for fname in all_files if fnmatch(fname, glob_pattern)]

        # First check if the expected file exists in the root of the model directory
        # If it doesn't, check if it exists in the backend subfolder.
        # If it does, set the subfolder to include the backend
        export = primary_full_path not in model_file_names
        if export and "subfolder" not in model_args:
            export = secondary_full_path not in model_file_names
            if not export:
                if len(model_file_names) > 1 and "file_name" not in model_args:
                    logger.warning(
                        f"Multiple {backend_name} files found in {load_path.as_posix()!r}: {model_file_names}, defaulting to {secondary_full_path!r}. "
                        f'Please specify the desired file name via `model_kwargs={{"file_name": "<file_name>"}}`.'
                    )
                model_args["subfolder"] = self.backend
                model_args["file_name"] = file_name

        # If the file_name contains subfolders, set it as the subfolder instead
        file_name_parts = Path(file_name).parts
        if len(file_name_parts) > 1:
            model_args["file_name"] = file_name_parts[-1]
            model_args["subfolder"] = Path(model_args.get("subfolder", ""), *file_name_parts[:-1]).as_posix()

        if export:
            logger.warning(
                f"No {file_name!r} found in {load_path.as_posix()!r}. Exporting the model to {backend_name}."
            )
            if model_file_names:
                logger.warning(
                    f"If you intended to load one of the {model_file_names} {backend_name} files, "
                    f'please specify the desired file name via `model_kwargs={{"file_name": "{model_file_names[0]}"}}`.'
                )

        return export, model_args

    def _backend_warn_to_save(self, model_name_or_path: str, is_local: str, backend_name: str) -> None:
        to_log = f"Saving the exported {backend_name} model is heavily recommended to avoid having to export it again."
        if is_local:
            to_log += f" Do so with `model.save_pretrained({model_name_or_path!r})`."
        else:
            to_log += f" Do so with `model.push_to_hub({model_name_or_path!r}, create_pr=True)`."
        logger.warning(to_log)

    def _load_t5_model(self, model_name_or_path, config, cache_dir, **model_args) -> None:
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel

        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )

    def _load_mt5_model(self, model_name_or_path, config, cache_dir, **model_args) -> None:
        """Loads the encoder model from T5"""
        from transformers import MT5EncoderModel

        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = MT5EncoderModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )

    def __repr__(self) -> str:
        return f"Transformer({self.get_config_dict()}) with Transformer model: {self.auto_model.__class__.__name__} "

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Returns token_embeddings, cls_token"""
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = self.auto_model(**trans_features, **kwargs, return_dict=False)
        output_tokens = output_states[0]

        # If the AutoModel is wrapped with a PeftModelForFeatureExtraction, then it may have added virtual tokens
        # We need to extend the attention mask to include these virtual tokens, or the pooling will fail
        if is_peft_available():
            from peft import PeftModelForFeatureExtraction

            if (
                isinstance(self.auto_model, PeftModelForFeatureExtraction)
                and self.auto_model.active_peft_config.is_prompt_learning
            ):
                batch_size = output_tokens.size(0)
                attention_mask = features["attention_mask"]
                prefix_attention_mask = torch.ones(
                    batch_size, self.auto_model.active_peft_config.num_virtual_tokens, device=attention_mask.device
                )
                features["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        features["token_embeddings"] = output_tokens

        if self.auto_model.config.output_hidden_states and len(output_states) > 2:
            all_layer_idx = 2  # I.e. after last_hidden_states and pooler_output
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features["all_layer_embeddings"] = hidden_states

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(
        self, texts: list[str] | list[dict] | list[tuple[str, str]], padding: str | bool = True
    ) -> dict[str, torch.Tensor]:
        """Tokenizes a text and maps tokens to token-ids"""
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length,
            )
        )
        return output

    def get_config_dict(self) -> dict[str, Any]:
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        self.auto_model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, "sentence_bert_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @classmethod
    def load(cls, input_path: str):
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        # Don't allow configs to set trust_remote_code
        if "model_args" in config and "trust_remote_code" in config["model_args"]:
            config["model_args"].pop("trust_remote_code")
        if "tokenizer_args" in config and "trust_remote_code" in config["tokenizer_args"]:
            config["tokenizer_args"].pop("trust_remote_code")
        if "config_args" in config and "trust_remote_code" in config["config_args"]:
            config["config_args"].pop("trust_remote_code")
        return cls(model_name_or_path=input_path, **config)

# %%
NUM_PROC = 16
print(NUM_PROC)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    pl.set_random_seed(seed)

seed_everything(SEED)

# %%
# /content/drive/MyDrive/kaggle-eedi/input/train_5folds_with_llm_infer.csv
df = pd.read_csv(f"{DATA_PATH}/train_5folds_with_llm_infer.csv")
df['is_synthetic'] = False
print(df.shape)

# %%
# 洪さんLLM synthetic_data
df_synth = pd.read_csv(f"{DATA_PATH}/synthetic_questions_render_with_answer_render_v1.csv")
df_synth = df_synth[~df_synth.isna().any(axis=1)].reset_index(drop=True)
df_synth = df_synth[df_synth["quality-gpt4o-mini"] > 2].reset_index(drop=True)
# df_synth = df_synth.sample(n=4000, random_state=0).reset_index(drop=True)

# 3rd subject nameを利用
df_synth = df_synth.rename({"ThirdSubjectName": "SubjectName"}, axis=1)
df_synth = df_synth.rename({"MisconceptionName": "Misconception"}, axis=1)

df_synth["is_synthetic"] = True
df_synth["fold"] = -1
print(df_synth.shape)

# %%
# 洪さんLLM synthetic_data
df_gpt = pd.read_csv(f"{DATA_PATH}/gpt-4o-mini-q-a_v2_render_v1.csv")

# rename
df_gpt = df_gpt.rename({"ConstructName-qwen25-72b-instruct": "ConstructName"}, axis=1)
df_gpt = df_gpt.rename({"MisconceptionName": "Misconception"}, axis=1)

# Qualityで絞り込み
df_gpt = df_gpt[df_gpt["quality-gpt4o-mini"] > 2].reset_index(drop=True)

df_gpt["is_synthetic"] = True
df_gpt["fold"] = -2
print(df_gpt.shape)

# %%
# 洪さんLLM synthetic_data
df_synth2 = pd.read_csv(f"{DATA_PATH}/synthetic-round2-render.csv")
df_synth2 = df_synth2[~df_synth2.isna().any(axis=1)].reset_index(drop=True)

df_synth2 = df_synth2.rename({"ConstructName-qwen25-72b-instruct": "ConstructName"}, axis=1)
df_synth2 = df_synth2.rename({"MisconceptionName": "Misconception"}, axis=1)

df_synth2 = df_synth2[df_synth2["quality-gpt4o-mini"] > 2].reset_index(drop=True)

df_synth2["is_synthetic"] = True
df_synth2["fold"] = -3
print(df_synth2.shape)

# %%
# 洪さんLLM synthetic_data
df_synth3 = pd.read_csv(f"{DATA_PATH}/synthetic-round3-render.csv")
df_synth3 = df_synth3[~df_synth3.isna().any(axis=1)].reset_index(drop=True)

df_synth3 = df_synth3.rename({"MisconceptionName": "Misconception"}, axis=1)

df_synth3 = df_synth3[df_synth3["quality-gpt4o-mini"] > 2].reset_index(drop=True)

df_synth3["is_synthetic"] = True
df_synth3["fold"] = -4
print(df_synth3.shape)

# %%
df = pd.concat([df, df_synth, df_gpt, df_synth2, df_synth3], axis=0).reset_index(drop=True)
print(df.shape)

# %%
df.head()

# %%
df[["SubjectName", "ConstructName", "QuestionText", "CorrectAnswerText", "AnswerText", "Misconception"]].isnull().sum(0)

# %%
def get_query_text(row):
    task_description = f'Given a math question and an incorrect answer, please retrieve the most relevant misconception causing the incorrect answer.'
    query_text = f"###Question###:{row['SubjectName']}-{row['ConstructName']}-{row['QuestionText']}\n###Correct Answer###:{row['CorrectAnswerText']}\n###Incorrect answer###:{row['AnswerText']}"
    return f'Instruct: {task_description}\nQuery: {query_text}'

# %%
df['InputText'] = df.apply(lambda x: get_query_text(x), axis=1)

# %%
print(df['InputText'].values[0])

# %%
df["InputText"].map(len).describe()

# %%
df = df[(df["InputText"].map(len) < 2000) | (~df['is_synthetic'])].reset_index(drop=True)
len(df)

# %%
df_mis = pd.read_csv(f"{DATA_PATH}/misconception_mapping.csv")
mis_map = df_mis.set_index("MisconceptionId")['a000-llama3-mega-misconception-aug-seed201_misunderstanding'].to_dict()
df['a000-llama3-mega-misconception-aug-seed201_misunderstanding'] = df['MisconceptionId'].map(mis_map)
df['Misconception'] = df['Misconception'] + ' ' + df['a000-llama3-mega-misconception-aug-seed201_misunderstanding']
df["Misconception"] = df["Misconception"].apply(lambda x: '\\boxed{' + x + '}')

# %%
df_not_synthetic = df[~df['is_synthetic']].reset_index(drop=True)
df_synthetic = df[df['is_synthetic']].reset_index(drop=True)
len(df_not_synthetic), len(df_synthetic)

# %%
def _sample_synthetic(df_synthetic):
    mis_idx_map = {}
    for i in range(len(df_synthetic)):
        row = df_synthetic.iloc[i]
        misconception_id = row['MisconceptionId']
        if misconception_id not in mis_idx_map:
            mis_idx_map[misconception_id] = []
        mis_idx_map[misconception_id].append(i)
    sampled_idxs = []
    for misconception_id, idx_list in mis_idx_map.items():
        sampled_idx = np.random.choice(idx_list, 1, replace=False)
        sampled_idxs.append(sampled_idx[0])
    sampled_df = df_synthetic.iloc[sampled_idxs].reset_index(drop=True)
    other_df = df_synthetic.drop(sampled_idxs).reset_index(drop=True)
    return sampled_df, other_df

def sample_synthetic(df_synthetic, num_synthetic):
    count = 0
    dfs = []
    while count < num_synthetic:
        sampled_df, other_df = _sample_synthetic(df_synthetic)
        dfs.append(sampled_df)
        count += len(sampled_df)
        df_synthetic = other_df
    return pd.concat(dfs, axis=0).reset_index(drop=True)

# %%
df_synthetic_sampled = sample_synthetic(df_synthetic, 40000)
len(df_synthetic_sampled)

# %%
df = pd.concat([df_not_synthetic, df_synthetic_sampled], axis=0).reset_index(drop=True)
len(df)

# %% [markdown]
# # Dataset

# %%
# df_train = df[df["fold"] != 0].reset_index(drop=True)
# df_test = df[df["fold"] == 0].reset_index()
df_train = df

# %%
train_ds = Dataset.from_pandas(df_train)
# test_ds = Dataset.from_pandas(df_test)

# %%
# ir_queries = dict(zip(df_test["index"], df_test["InputText"]))

# %%
df_courpus = pd.read_csv(f"{DATA_PATH}/misconception_mapping.csv")
df_courpus["MisconceptionName"] = df_courpus["MisconceptionName"] + ' ' + df_courpus['a000-llama3-mega-misconception-aug-seed201_misunderstanding']
df_courpus["MisconceptionName"] = df_courpus["MisconceptionName"].apply(lambda x: '\\boxed{' + x + '}')
ir_corpus = df_courpus[["MisconceptionId", "MisconceptionName"]].drop_duplicates(['MisconceptionId']).reset_index(drop=True)
ir_corpus = dict(zip(ir_corpus.MisconceptionId, ir_corpus.MisconceptionName))

# %%
print(df_courpus["MisconceptionName"].values[0])

# %%
# df_test_agg = df_test.groupby(['index'])['MisconceptionId'].agg(set).reset_index()

# %%
# ir_relevant_docs = dict(zip(df_test_agg["index"], df_test_agg["MisconceptionId"]))

# %% [markdown]
# # Model

# %%
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

# %%
base_model = LLM2Vec.from_pretrained(
    MODEL_NAME,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    use_cache=False,
    #attn_implementation="flash_attention_2"
)

# %%
base_tokenizer = base_model.tokenizer

# %%
base_model.model

# %%
config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="FEATURE_EXTRACTION",
)

# %%
base_model = prepare_model_for_kbit_training(base_model.model)
print('model loaded')
base_model = get_peft_model(base_model, config)
# base_model = get_peft_model(base_model.model, config)
base_model.print_trainable_parameters()

# %%
model = SentenceTransformer("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

# %%
model._first_module().tokenizer

# %%
model._first_module().auto_model = base_model
model._first_module().tokenizer = base_tokenizer

# %%
model[1].pooling_mode_mean_tokens = False
model[1].pooling_mode_lasttoken = True

# %%
model

# %%
del base_model
torch.cuda.empty_cache()

# %% [markdown]
# # Training

# %%
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
# loss = MultipleNegativesRankingLoss(model)

# %%
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=MODEL_OUTPUT_PATH,
    # Optional training parameters:
    optim="paged_adamw_8bit",
    num_train_epochs=EPOCH,
    dataloader_num_workers=NUM_PROC,
    per_device_train_batch_size=BS,
#    gradient_accumulation_steps = 2,
#    per_device_eval_batch_size=BS,
    # learning_rate=LR,
    warmup_ratio=0.0,
    fp16=False,
    bf16=True,
    # batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    # lr_scheduler_type="cosine_with_restarts",
#    eval_strategy="epoch",
#    eval_steps=8,
    save_strategy="epoch",
    save_steps=1,
    save_total_limit=10,
    logging_steps=1000,
    report_to=REPORT_TO,  # Will be used in W&B if `wandb` is installed
#    metric_for_best_model="eval_cosine_map@25", # eval_cosine_recall@25
    do_eval=False,
    push_to_hub=False,
#    load_best_model_at_end=True,
    # gradient_checkpointing_kwargs=True
)

# %%
# dev_evaluator = InformationRetrievalEvaluator(
#    ir_queries, ir_corpus, ir_relevant_docs,
#    accuracy_at_k=[25],
#    precision_recall_at_k=[25, 50, 100],
#    map_at_k=[25])

# %%
trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds.select_columns(
            ["InputText", "Misconception"]
        ),
#        eval_dataset=test_ds.select_columns(
#            ["InputText", "Misconception"]
#        ),
        loss=loss,
#        evaluator=dev_evaluator,
    )

# %%
trainer.train()

# %%



