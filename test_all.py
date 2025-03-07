from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import sys

BASE_MODEL=""

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,device_map="auto", torch_dtype=torch.bfloat16).eval()

prompt = """
### As a query spelling error correction model, your task is to automatically detect and correct query spelling errors in the query. If the query does not contain errors, output the original query.
The input query is:
"""
input_flile = sys.argv[1]
with open(input_flile + ".finetune_3b", "w") as fp_out:
    with open(input_flile)as fp:
        text_list = []
        query_list = []
        for line in fp:
            query = line.strip("\n").split("\t")[0]
            messages = [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt + query + "\n ### Out query: "}
            ]
            text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            text_list.append(text)
            query_list.append(query)
            if len(text_list) == 512:
                model_inputs = tokenizer(text_list, return_tensors="pt", padding=True, padding_side="left").to(model.device)
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for text, response in zip(query_list, responses):
                    fp_out.write(text + "\t" + response.replace("\n","") + "\n")
                text_list.clear()
                query_list.clear()
    if len(text_list) > 0:
        model_inputs = tokenizer(text_list, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for text, response in zip(query_list, responses):
            fp_out.write(text + "\t" + response.replace("\n","") + "\n")
        text_list.clear()
        query_list.clear()