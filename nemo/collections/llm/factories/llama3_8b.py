import pytorch_lightning as pl

from nemo import lightning as nl
from nemo.collections.llm.utils import Config, factory
from nemo.collections.llm.gpt.model.llama import Llama3Config8B, LlamaModel


@factory
def llama3_8b() -> pl.LightningModule:
    return LlamaModel(Llama3Config8B())


# Ideally it should become:
# @factory(name="llama3_8b")
# def model() -> pl.LightningModule:
#     return LlamaModel(Llama3Config8B())


@factory
def llama3_8b_strategy(seq_length=8192) -> Config[nl.MegatronStrategy]:
    if seq_length <= 8192:
        return Config(nl.MegatronStrategy, tensor_model_parallel_size=2)
    elif seq_length <= 16384:
        return llama3_8b_strategy_context16k()
    elif seq_length <= 65536:
        return llama3_8b_strategy_context64k()
    else:
        raise ValueError(f"Unsupported sequence length: {seq_length}")
    

@factory
def llama3_8b_strategy_context16k() -> Config[nl.MegatronStrategy]:
    return Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=4,
        context_parallel_size=2,
        sequence_parallel=True,
        expert_model_parallel_size=1,
    )
    

@factory
def llama3_8b_strategy_context64k() -> Config[nl.MegatronStrategy]:
    return Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=8,
        context_parallel_size=4,
        sequence_parallel=True,
        expert_model_parallel_size=1,
    )


@factory
def llama3_8b_trainer(devices=8, seq_length=8192) -> Config[nl.Trainer]:
    return Config(
        nl.Trainer,
        devices=devices,
        max_steps=100,
        accelerator="gpu",
        strategy=llama3_8b_strategy(seq_length=seq_length),
        plugins=Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
    )
    

@factory
def llama3_8b_hf_resume() -> nl.AutoResume:
    return nl.AutoResume(import_path="hf://meta-llama/Meta-Llama-3-8B")


# TODO: Fix the name-arg inside the factory-function so we don't need to do this
model = llama3_8b
strategy = llama3_8b_strategy
strategy_context16k = llama3_8b_strategy_context16k
strategy_context64k = llama3_8b_strategy_context64k
trainer = llama3_8b_trainer
resume_hf = llama3_8b_hf_resume
