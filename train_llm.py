import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM, ModelConfig, TrlParser, ScriptArguments

def gen_from_jsonl(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip("\n").split("\t")
            yield {"text": line[0], "corrected_text": line[1]}

template = """
As a query spelling error correction model, your task is to automatically detect and correct query spelling errors in the query. If the query does not contain errors, output the original query.
The input query is:
"""

def make_map_fn(split):
    def _map_fn(example, idx):
        example['instruction'] = example['text']
        example['output'] = example['corrected_text']
        return example
    return _map_fn




def formatting_prompts_func(example):
    output_texts = []
    # print(len(example['instruction']))
    for i in range(len(example['instruction'])):
        # print(example['instruction'][i])
        try:
            text = f"### {template} {example['instruction'][i]}\n ### Out query: {example['output'][i]}"
            output_texts.append(text)
        except:
            continue
    return output_texts


def main(training_args, model_args):
    model_name_or_path = ""
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': "qspell_250k_train.txt"})
    dataset = raw_dataset.map(function=make_map_fn('train'), with_indices=True)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    response_template = " ### Out query:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(training_args, model_args)