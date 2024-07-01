from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch
import time
import os

os.environ['HUGGINGFACE_HUB_CACHE'] = "/scratch/dshah47/.cache/"

def load_model(llm):
    model_id = os.environ['HUGGINGFACE_HUB_CACHE'] + 'licensed_models/' + llm
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=True
    )
    return(model, tokenizer)

def setup_inputs(question, tokenizer):
    system_prompt = "You are an expert asked to classify text from a patent according to why the inventor may have used a scientific article in their invention. I will give you a chunk of text from a patent, and a reference that appears in the chunk. You will respond with ONE WORD ONLY: return 'Tool' if you believe the reference was used by a tool or technique by the inventor to make the invention, or 'Background' if it is general background knowledge, an unsolved problem the invention will tackle, a reason the invention was created, or a potential use case. Remember, reply ONLY with Tool or Background."
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Text:{question}<|eot_id|><|start_header_id|>Expert: <|end_header_id|>"

    inputs = tokenizer([prompt],
                   return_tensors="pt")
    return(inputs)


def get_prob(word, score):
    if word.lower() == 'tool':
        p = np.exp(score.cpu().numpy())
    elif word.lower() == 'background':
        p = 1-np.exp(score.cpu().numpy())
    else:
        p = np.nan
    return(p)


def next_token(model, tokenizer, question):
    inputs = setup_inputs(question, tokenizer)
    inputs.to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=2,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        kwargs={"temperature": 0},
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )
    transition_scores = model.compute_transition_scores(
        output.sequences,
        output.scores,
        normalize_logits=True
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        token_str = tokenizer.decode(tok).split()
        if not token_str:
            continue
        token_str = token_str[0]
        if token_str in ["<|start_header_id|>", "<|eot_id|>"] or not token_str:
            continue
    return(get_prob(token_str, score))


def classify_patent(model, tokenizer, question, reps):
    probs = np.empty(reps)
    probs.fill(np.nan)
    for rep in range(reps):
        probs[rep] = next_token(model, tokenizer, question)

    #if np.all(np.isnan(probs)):
        #return np.nan

    return(np.round(np.nanmedian(probs), decimals=2))

def main(csv_file, llm, reps):
    patent_data = pd.read_csv(csv_file,
                              encoding="utf-8")

    model, tokenizer = load_model(llm)

    for i in tqdm(range(len(patent_data.index))):
        patent_data.loc[i, 'probability'] = classify_patent(model,
                                                       tokenizer,
                                                       patent_data.iloc[i,1],
                                                       reps)
        if i % 20 == 0:
            patent_data.to_csv("results/llama3-" + os.path.basename(csv_file),
                    index=False)
    patent_data.to_csv("results/llama3-" + os.path.basename(csv_file),
                      index=False)
                                                  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            "Patent text classification using an LLM")
    parser.add_argument("csv_file", help="CSV file with data", type=str)
    parser.add_argument("llm", help="LLM", type=str)
    parser.add_argument("reps", help="Repetitions to validate each answer",
            type=int, nargs='?', default=4)
    args = parser.parse_args()

    start = time.time()
    main(args.csv_file, args.llm, args.reps)
    end = time.time()
    print(f"Time elapsed -- {end - start}")

