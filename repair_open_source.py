from __future__ import absolute_import, division, print_function
import os
import re
import json
import torch
import random
import argparse
import warnings
import numpy as np
from model import Model
from sklearn.cluster import KMeans
from utils.build_d4j import build_d4j1_2

from utils.parse_d4j import clean_parse_d4j
from utils.parse_rwb import clean_parse_rwb

from utils.validate_d4j import validate_patch
from utils.validate_rwb import validate_patch_rwb

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['STARCODER_AUTH_TOKEN'] = ""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_validation(args, file, output):
    pattern = r'(?s)// Fixed Function.*?// End'
    results = re.findall(pattern, output, re.DOTALL)
    if len(results) >= 1:
        patch = results[0].replace("// Fixed Function", "").replace("// End", "").strip()
        patch = patch[:patch.rfind("}")+1]
    else:
        return False, None
    try:
        with open(f"./Results/{args.dataset}/{file}", 'w') as f:
            f.write(patch)
    except:
        with open(f"./Results/{args.dataset}/{file}", 'w') as f:
            f.write("write error ... ")
    return validate_patch(f"./Results/{args.dataset}/{file}", patch, "./Datasets/D4J/location")


def run_validation_rwb(args, file, output):
    pattern = r'(?s)// Fixed Function.*?// End'
    results = re.findall(pattern, output, re.DOTALL)
    if len(results) >= 1:
        patch = results[0].replace("// Fixed Function", "").replace("// End", "").strip()
        patch = patch[:patch.rfind("}")+1]
    else:
        return False, None
    try:
        with open(f"./Results/{args.dataset}/{file}", 'w') as f:
            f.write(patch)
    except:
        with open(f"./Results/{args.dataset}/{file}", 'w') as f:
            f.write("write error ... ")
    return validate_patch_rwb(f"./Results/{args.dataset}/{file}")


def get_embedding(args, model, tokenizer, think_dataset):
    with open(f"./Results/{args.model}/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r') as f:
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
        with open(f"./Results/{args.model}/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'w') as f:
            json.dump(cot, f, indent=4)


def get_embedding_UniXcoder(args, model, tokenizer, think_dataset):
    with open(f"./Results/{args.model}/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r') as f:
        cot = json.load(f)
    for file, repair_result in cot.items():
        if repair_result[-1]["valid"] == False or "embedding_UniXcoder" in repair_result[-1].keys(): continue
        buggy = think_dataset[file]["buggy"]
        tokenized_code = tokenizer.encode_plus(buggy, max_length=400, return_tensors="pt")
        outputs = model(**tokenized_code)    
        cot[file][-1]["embedding_UniXcoder"] = outputs[0][0, 0, :].detach().numpy().tolist()
        with open(f"./Results/{args.model}/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'w') as f:
            json.dump(cot, f, indent=4)


def get_clusters(args):
    if args.select == "CSelect":
        embedding = "embedding"
    else:
        embedding = "embedding_UniXcoder"
    cot = json.load(open(f"./Results/{args.model}/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r'))
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
    cot = json.load(open(f"./Results/{args.model}/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r'))
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


def generate_chain_of_thought(args, think_dataset, tokenizer, model):
    with open(f"./Datasets/D4J/generate_prompt{'_pfl' if args.pfl else ''}_open_source.txt") as f:
        generate_prompt = f.read()

    try:
        with open(f"./Results/{args.model}/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r') as f:
            cot = json.load(f)
    except:
        cot = {}

    for file, bug in tqdm(think_dataset.items()):
        if file in cot.keys(): continue
        print("Repairing bug {} ... ".format(file.split(".")[0]))
        
        if args.pfl:
            buggy = add_bug_comments(bug["buggy"], bug["location"])
            prompt = generate_prompt.format(bug=buggy)
        else:
            buggy = bug['buggy']
            prompt = generate_prompt.format(bug=buggy)

        if args.model == 'CodeLlama':
            prompt = f"<s>[INST] <<SYS>>\nYou are an Automatic Program Repair Tool.\n<</SYS>>\n\n{prompt} [/INST]" 
        elif args.model == 'DeepSeek-Coder':
            prompt = "You are an Automatic Program Repair Tool.\n### Instruction:\n" + prompt
            
        for _ in range(args.sample): 
            repair_result = []
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            buggy_length = tokenizer.encode(buggy, return_tensors="pt").to(model.device).shape[1]
            total_input_tokens = inputs.shape[1]
            model_max_length = 4096
            if total_input_tokens >= model_max_length:
                repair_result.append({"output": "# Token size exceeded.", "valid": False})
                break
            max_new_tokens = buggy_length+100
            
            raw_outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=args.p,
                top_k=args.k,
                temperature=args.temperature,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
             
            output = tokenizer.decode(raw_outputs[0][len(inputs[0]): ])
            
            valid, _ = run_validation(args, file, output)
            output = output[: output.find("// End")].strip()
            repair_result.append({"output": output, "valid": valid})
            if valid: break
        cot[file] = repair_result
        with open(f"./Results/{args.model}/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'w') as f:
            json.dump(cot, f, indent=4)


def chain_of_thought_repair(args, Select_model, Select_tokenizer, think_dataset, inference_files, tokenizer, model):
    with open(f"./Datasets/D4J/repair_prompt{'_pfl' if args.pfl else ''}_open_source.txt") as f:
        repair_prompt = f.read()

    with open(f"./Results/{args.model}/{args.dataset}/cot{'_pfl' if args.pfl else ''}.json", 'r') as f:
        cot = json.load(f)

    try:
        with open(f"./Results/{args.model}/{args.dataset}/repair{'_pfl' if args.pfl else ''}.json", 'r') as f:
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
            origin_prompt = repair_prompt.format(example_bug_1=example_buggy_1, example_cot_1=cot[examples[0]][-1]["output"], example_bug_2=example_buggy_2, example_cot_2=cot[examples[1]][-1]["output"], bug=buggy)
        else:
            buggy = bug['buggy']
            origin_prompt = repair_prompt.format(example_bug_1=think_dataset[examples[0]]["buggy"], example_cot_1=cot[examples[0]][-1]["output"], example_bug_2=think_dataset[examples[1]]["buggy"], example_cot_2=cot[examples[1]][-1]["output"], bug=buggy)

        if file in repair.keys():
            repair_results = repair[file]
        else:
            repair_results = []

        for _ in range(args.sample - len(repair_results)):
            prompt = origin_prompt
            if args.model == 'CodeLlama':
                prompt = f"<s>[INST] <<SYS>>\nYou are an Automatic Program Repair Tool.\n<</SYS>>\n\n{origin_prompt} [/INST]" 
            elif args.model == 'DeepSeek-Coder':
                prompt = "You are an Automatic Program Repair Tool.\n### Instruction:\n" + origin_prompt
            repair_result = []
            for _ in range(args.chance):
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                buggy_length = tokenizer.encode(buggy, return_tensors="pt").to(model.device).shape[1]
                total_input_tokens = inputs.shape[1]
                model_max_length = 4096
                if total_input_tokens + buggy_length + 100 >= model_max_length:
                    repair_result.append({"output": "# Token size exceeded.", "valid": False})
                    break
                max_new_tokens = buggy_length+100
                
                raw_outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=args.p,
                    top_k=args.k,
                    temperature=args.temperature,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

                output = tokenizer.decode(raw_outputs[0][len(inputs[0]): ])
                if args.dataset == "RWBV2.0":
                    valid, message = run_validation_rwb(args, file, output)
                else:
                    valid, message = run_validation(args, file, output)
                output = output[:output.find("// End")].strip()
                output = output[:output.rfind("}")+1]
                if args.model == 'StarCoder':
                    prompt = f"{prompt}{output}\n// End\nThe fixed version is still not correct.\n{message}\nPlease fix it again. Keep the original '// Fixed Function' and '// End' formats. Let's think step by step. "
                elif args.model == 'CodeLlama':
                    prompt = f"{prompt}{output}\n// End </s><s>[INST]The fixed version is still not correct.\n{message}\nPlease fix it again. Keep the original '// Fixed Function' and '// End' formats. Let's think step by step. [/INST]" 
                elif args.model == 'DeepSeek-Coder':
                    prompt = f"{prompt}\n### Response:\n{output}\n// End\n<|EOT|>\n### Instruction:\nThe fixed version is still not correct.\n{message}\nPlease fix it again. Keep the original '// Fixed Function' and '// End' formats. Let's think step by step. "
                repair_result.append({"output": output, "valid": valid})
                if valid: break
            repair_results.append(repair_result)
            repair[file] = repair_results
            with open(f"./Results/{args.model}/{args.dataset}/repair{'_pfl' if args.pfl else ''}.json", 'w') as f:
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
                        help="Dataset to use, current support: D4JV1.2, D4JV2.0, RWBV2.0")
    parser.add_argument("--select", type=str, default="CSelect",
                        help="Selection strategy to use, current support: CSelect, SSelect, RSelect")
    parser.add_argument("--pfl", type=bool, default=False)
    parser.add_argument('--model', help='model to use for code translation. should be one of [CodeGeeX,StarCoder,CodeGen,TB-Airoboros,TB-Vicuna,LLaMa,CodeLLama]', required=True, type=str)
    parser.add_argument("--n_example", type=int, default=2)
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--chance", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--k', help='The number of highest probability vocabulary tokens to keep for top-k-filtering. Only applies for sampling mode, with range from 1 to 100.', type=int, default=50)
    parser.add_argument('--p', help='Only the most probable tokens with probabilities that add up to top_p or higher are considered during decoding. The valid range is 0.0 to 1.0. 1.0 is equivalent to disabled and is the default. Only applies to sampling mode. Also known as nucleus sampling.', type=float, default=0.95)
    parser.add_argument('--temperature', help='A value used to warp next-token probabilities in sampling mode. Values less than 1.0 sharpen the probability distribution, resulting in "less random" output. Values greater than 1.0 flatten the probability distribution, resulting in "more random" output. A value of 1.0 has no effect and is the default. The allowed range is 0.0 to 2.0.', type=float, default=1)
    
    # model
    parser.add_argument("--model_dir", default="saved_models", type=str)
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base-nine", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=400, type=int,
                        help="Optional input sequence length after tokenization.")         
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    tokenizer, model = None, None
    if args.model == 'StarCoder':
        tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoder', use_auth_token=os.environ['STARCODER_AUTH_TOKEN'], cache_dir='./huggingface', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('bigcode/starcoder', use_auth_token=os.environ['STARCODER_AUTH_TOKEN'], cache_dir='./huggingface', trust_remote_code=True, device_map='auto')
    elif args.model == 'CodeLlama':
        tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-13b-Instruct-hf', cache_dir='./huggingface', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('codellama/CodeLlama-13b-Instruct-hf', cache_dir='./huggingface', trust_remote_code=True, device_map='auto')
    elif args.model == 'DeepSeek-Coder':
        tokenizer = AutoTokenizer.from_pretrained('./deepseek-coder-6.7b-instruct', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('./deepseek-coder-6.7-instruct', trust_remote_code=True, device_map='auto')

    out_folder = f'./Results/{args.model}/{args.dataset}'
    os.makedirs(out_folder, exist_ok=True)
    
    d4j_dataset = clean_parse_d4j(folder="./Datasets/")

    d4j1_2 = build_d4j1_2()
    
    if args.dataset == "D4JV1.2":
        think_dataset = {key: value for key, value in d4j_dataset.items() if key.split('.')[0] not in d4j1_2}
        inference_dataset = {key: value for key, value in d4j_dataset.items() if key.split('.')[0] in d4j1_2}
    elif args.dataset == "D4JV2.0":
        think_dataset = {key: value for key, value in d4j_dataset.items() if key.split('.')[0] in d4j1_2}
        inference_dataset = {key: value for key, value in d4j_dataset.items() if key.split('.')[0] not in d4j1_2}
    elif args.dataset == "RWBV2.0":
        think_dataset = d4j_dataset
        inference_dataset = clean_parse_rwb(folder="./Datasets/")

    generate_chain_of_thought(args, think_dataset, tokenizer, model)

    if args.select == "CSelect":
        Select_model, Select_tokenizer = load_model(args)
        get_embedding(args, Select_model, Select_tokenizer, think_dataset)
    else:
        Select_tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
        Select_model = AutoModel.from_pretrained("microsoft/unixcoder-base-nine")
        get_embedding_UniXcoder(args, Select_model, Select_tokenizer, think_dataset)

    chain_of_thought_repair(args, Select_model, Select_tokenizer, think_dataset, inference_dataset, tokenizer, model)


if __name__=="__main__":
    main()