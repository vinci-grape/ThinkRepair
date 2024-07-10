from __future__ import absolute_import, division, print_function
import os
import time
import json
import torch
import random
import argparse
import tiktoken
import warnings
import numpy as np
from model import Model
from openai import OpenAI
from sklearn.cluster import KMeans
from utils.build_d4j import build_d4j1_2

from utils.parse_d4j import clean_parse_d4j
from utils.parse_rwb import clean_parse_rwb

from utils.validate_d4j import validate_patch
from utils.validate_rwb import validate_patch_rwb

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoConfig

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

encoding_name = 'cl100k_base'
encoding = tiktoken.get_encoding(encoding_name)

client = OpenAI(api_key="")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def chatgpt_encoding_count(input: str) -> int:
    """ chatgpt token count """
    token_integers = encoding.encode(input)
    num_tokens = len(token_integers)
    return num_tokens


def run_validation(args, file, output):
    start = output.find("// Fixed Function")
    end = output.rfind("}") + 1
    patch = output[start:end]
    try:
        with open(f"./Results/ChatGPT/{args.dataset}/{file}", 'w') as f:
            f.write(patch)
    except:
        with open(f"./Results/ChatGPT/{args.dataset}/{file}", 'w') as f:
            f.write("write error ... ")
    return validate_patch(f"./Results/ChatGPT/{args.dataset}/{file}", patch, "./Datasets/D4J/location")


def run_validation_rwb(args, file, output):
    start = output.find("// Fixed Function")
    end = output.rfind("}") + 1
    patch = output[start:end]
    try:
        with open(f"./Results/ChatGPT/{args.dataset}/{file}", 'w') as f:
            f.write(patch)
    except:
        with open(f"./Results/ChatGPT/{args.dataset}/{file}", 'w') as f:
            f.write("write error ... ")
    return validate_patch_rwb(f"./Results/ChatGPT/{args.dataset}/{file}", args.dataset)


def request_engine(messages):
    ret = None
    while ret is None:
        try:
            ret = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=1,
                messages=messages
            )
        except Exception as e:
            print(e)
            return None
    return ret


def get_embedding(args, model, tokenizer, think_dataset):
    with open(f"./Results/ChatGPT/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r') as f:
        cot = json.load(f)
    for file, repair_result in cot.items():
        if repair_result[-1]["valid"] == False or "embedding" in repair_result[-1].keys(): continue

        buggy = think_dataset[file]["buggy"]
        code_tokens = tokenizer.tokenize(buggy)[:args.block_size-4]
        source_tokens = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        cot[file][-1]["embedding"] = model.get_xcode_vec(torch.tensor(source_ids).unsqueeze(0).to(args.device))[0].cpu().detach().numpy().tolist()
        with open(f"./Results/ChatGPT/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'w') as f:
            json.dump(cot, f, indent=4)


def get_embedding_UniXcoder(args, model, tokenizer, think_dataset):
    with open(f"./Results/ChatGPT/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r') as f:
        cot = json.load(f)
    for file, repair_result in cot.items():
        if repair_result[-1]["valid"] == False or "embedding_UniXcoder" in repair_result[-1].keys(): continue
        buggy = think_dataset[file]["buggy"]
        tokenized_code = tokenizer.encode_plus(buggy, max_length=400, return_tensors="pt")
        outputs = model(**tokenized_code)    
        cot[file][-1]["embedding_UniXcoder"] = outputs[0][0, 0, :].detach().numpy().tolist()
        with open(f"./Results/ChatGPT/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'w') as f:
            json.dump(cot, f, indent=4)


def get_clusters(args):
    if args.select == "CSelect":
        embedding = "embedding"
    else:
        embedding = "embedding_UniXcoder"
    cot = json.load(open(f"./Results/ChatGPT/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r'))
    embeddings = np.asarray([repair_result[-1][embedding] for _, repair_result in cot.items() if repair_result[-1]["valid"]])
    kmeans = KMeans(n_clusters=args.n_example, n_init=10, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    return labels
    

def get_example_from_clusters(args, model, tokenizer, buggy, labels):
    print("Get buggy func embedding")
    if args.select == "CSelect":
        embedding = "embedding"
    else:
        embedding = "embedding_UniXcoder"
    cot = json.load(open(f"./Results/ChatGPT/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r'))
    embeddings = np.asarray([repair_result[-1][embedding] for _, repair_result in cot.items() if repair_result[-1]["valid"]])
    points = [file for file, repair_result in cot.items() if repair_result[-1]["valid"]]

    selected_points = []
    if args.select == "CSelect":
        code_tokens = tokenizer.tokenize(buggy)[:args.block_size-4]
        source_tokens = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        buggy_embedding = model.get_xcode_vec(torch.tensor(source_ids).unsqueeze(0).to(args.device))[0].cpu().detach().numpy().tolist()
    else:
        tokenized_code = tokenizer.encode_plus(buggy, max_length=400, return_tensors="pt")
        outputs = model(**tokenized_code)    
        buggy_embedding = outputs[0][0, 0, :].detach().numpy().tolist()

    for i in range(args.n_example):
        cluster_points = np.where(labels == i)[0]
        cluster_embeddings = [embeddings[point] for point in cluster_points]
        similarities = cosine_similarity([buggy_embedding], cluster_embeddings)
        most_similar_index = np.argmax(similarities)
        selected_point = cluster_points[most_similar_index]
        selected_points.append(points[selected_point])
    return selected_points


def add_bug_comments(code_string, buggy_line_numbers):
    lines = code_string.split("\n")
    for line_number in buggy_line_numbers:
        if 1 <= line_number <= len(lines):
            lines[line_number-1] += " // Buggy Line"
    modified_code_string = "\n".join(lines)
    return modified_code_string


def generate_chain_of_thought(args, think_dataset):
    with open(f"./Datasets/D4J/generate_prompt{'_pfl' if args.pfl else ''}.txt") as f:
        generate_prompt = f.read()

    try:
        with open(f"./Results/ChatGPT/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r') as f:
            cot = json.load(f)
    except:
        cot = {}

    for file, bug in think_dataset.items():
        if file in cot.keys(): continue
        print("Repairing bug {} ... ".format(file.split(".")[0]))
        
        if args.pfl:
            buggy_func = bug["buggy"]
            buggy_lines = bug["location"]
            modified_func = add_bug_comments(buggy_func, buggy_lines)
            prompt = generate_prompt.format(bug=modified_func)
        else:
            prompt = generate_prompt.format(bug=bug['buggy'])

        for _ in range(args.sample):
            repair_result = []
            messages = [
                        {"role": "system", "content": "You are an Automatic Program Repair Tool"}, 
                        {"role": "user", "content": prompt}
                    ]
            if chatgpt_encoding_count(str(messages)) >= 4096: break
            output = request_engine(messages).choices[0].message.content
            valid, _ = run_validation(args, file, output)
            end = output.rfind("}") + 1
            repair_result.append({"output": output[: end], "valid": valid})
            if valid: break
        cot[file] = repair_result
        with open(f"./Results/ChatGPT/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'w') as f:
            json.dump(cot, f, indent=4)


def chain_of_thought_repair(args, Select_model, Select_tokenizer, think_dataset, inference_files):
    with open(f"./Datasets/D4J/repair_prompt{'_pfl' if args.pfl else ''}.txt") as f:
        repair_prompt = f.read()

    with open(f"./Results/ChatGPT/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r') as f:
        cot = json.load(f)

    try:
        with open(f"./Results/ChatGPT/{args.dataset}/repair{'_pfl' if args.pfl else ''}.json", 'r') as f:
            repair = json.load(f)
    except:
        repair = {}

    labels = get_clusters(args)
    for file, bug in inference_files.items():
        print("Repairing bug {} ... ".format(file.split(".")[0]))

        if args.select != "RSelect":
            examples = get_example_from_clusters(args, Select_model, Select_tokenizer, bug['buggy'], labels)
        else:
            examples = np.random.choice(list(cot.keys()), args.n_example)

        if args.pfl:
            buggy = add_bug_comments(bug["buggy"], bug["location"])
            example_buggy_1 = add_bug_comments(think_dataset[examples[0]]["buggy"], think_dataset[examples[0]]["location"])
            example_buggy_2 = add_bug_comments(think_dataset[examples[1]]["buggy"], think_dataset[examples[1]]["location"])
            prompt = repair_prompt.format(example_bug_1=example_buggy_1, example_cot_1=cot[examples[0]][-1]["output"], example_bug_2=example_buggy_2, example_cot_2=cot[examples[1]][-1]["output"], bug=buggy)
        else:
            prompt = repair_prompt.format(example_bug_1=think_dataset[examples[0]]["buggy"], example_cot_1=cot[examples[0]][-1]["output"], example_bug_2=think_dataset[examples[1]]["buggy"], example_cot_2=cot[examples[1]][-1]["output"], bug=bug['buggy'])

        if file in repair.keys():
            repair_results = repair[file]
        else:
            repair_results = []

        for _ in range(args.sample - len(repair_results)):
            messages = [
                        {"role": "system", "content": "You are an Automatic Program Repair Tool."}, 
                        {"role": "user", "content": prompt}
                    ]
            repair_result = []
            for _ in range(args.chance):
                output = request_engine(messages).choices[0].message.content
                if args.dataset == "RWBV1.0":
                    valid, message = run_validation_rwb(args, file, output)
                else:
                    valid, message = run_validation(args, file, output)
                messages.append({"role": "assistant", "content": output})
                print(f"message: {message}")
                messages.append({"role": "user", "content": f"The fixed version is still not correct.\n{message}\nPlease fix it again. Let's think step by step. "})
                repair_result.append({"output": output, "valid": valid})
                if valid or chatgpt_encoding_count(str(messages)) > 4096: 
                    break

            if valid or chatgpt_encoding_count(str(messages)) > 4096 or len(repair_result) == args.chance:
                repair_result[-1]["status"] = "finish"
                
            repair_results.append(repair_result)
            repair[file] = repair_results
            with open(f"./Results/ChatGPT/{args.dataset}/repair{'_pfl' if args.pfl else ''}.json", 'w') as f:
                json.dump(repair, f, indent=4)


def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path) 
    model = Model(model, config, tokenizer, args)
    model.to(device)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  

    checkpoint_prefix = 'checkpoint-best-loss/model.bin'
    model_dir = os.path.join(args.model_dir, '{}'.format(checkpoint_prefix))  
    model = torch.load(model_dir)
    
    return model, tokenizer    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="D4JV1.2",
                        help="Dataset to use, current support: D4JV1.2, D4JV2.0, RWBV1.0")
    parser.add_argument("--select", type=str, default="CSelect",
                        help="Selection strategy to use, current support: CSelect, SSelect, RSelect")
    parser.add_argument("--pfl", type=bool, default=True)
    parser.add_argument("--n_example", type=int, default=2)
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--chance", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # model
    parser.add_argument("--model_dir", default="saved_models", type=str)
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base-nine", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=400, type=int,
                        help="Optional input sequence length after tokenization.")
    
    args = parser.parse_args()

    out_folder = f'./Results/ChatGPT/{args.dataset}'
    os.makedirs(out_folder, exist_ok=True)

    d4j_dataset = clean_parse_d4j(folder="./Datasets/")

    set_seed(args.seed)

    d4j1_2 = build_d4j1_2()
    
    if args.dataset == "D4JV1.2":
        think_dataset = {key: value for key, value in d4j_dataset.items() if key.split('.')[0] not in d4j1_2}
        inference_dataset = {key: value for key, value in d4j_dataset.items() if key.split('.')[0] in d4j1_2}
    elif args.dataset == "D4JV2.0":
        think_dataset = {key: value for key, value in d4j_dataset.items() if key.split('.')[0] in d4j1_2}
        inference_dataset = {key: value for key, value in d4j_dataset.items() if key.split('.')[0] not in d4j1_2}
    elif args.dataset == "RWBV1.0":
        think_dataset = d4j_dataset
        inference_dataset = clean_parse_rwb(folder="./Datasets/", dataset=args.dataset)

    generate_chain_of_thought(args, think_dataset)

    if args.select == "CSelect":
        Select_model, Select_tokenizer = load_model(args)
        get_embedding(args, Select_model, Select_tokenizer, think_dataset)
    else:
        Select_tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
        Select_model = AutoModel.from_pretrained("microsoft/unixcoder-base-nine")
        get_embedding_UniXcoder(args, Select_model, Select_tokenizer, think_dataset)

    chain_of_thought_repair(args, Select_model, Select_tokenizer, think_dataset, inference_dataset)


if __name__=="__main__":
    main()