import json


def clean_parse_rwb(folder, dataset):
    if dataset == "RWBV1.0":
        with open(folder + "RWB/RWB-V1.0.json", "r") as f:
            result = json.load(f)
    else:
        with open(folder + "RWB/RWB-V2.0.json", "r") as f:
            result = json.load(f)
            
    cleaned_result = {}
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
        lines = v['fix'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"]["fix"] = "\n".join([line[leading_white_space:] for line in lines])
        cleaned_result[k + ".java"]["location"] = [location - v['start'] + 1 for location in v["location"]]
    return cleaned_result