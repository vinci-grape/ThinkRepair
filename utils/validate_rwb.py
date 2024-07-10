import json
import javalang
import subprocess
import re
import subprocess as sp


def git_reset(repo_dir_path):
    sp.run(['git', 'reset', '--hard', 'HEAD'],
           cwd=repo_dir_path, stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def git_clean(repo_dir_path):
    sp.run(['git', 'clean', '-df'],
           cwd=repo_dir_path, stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def compile_repo(repo_dir_path):
    # actual compiling
    compile_proc = sp.run(
        ['mvn', 'compile', '-Drat.skip=true'],
        stdout=sp.PIPE, stderr=sp.PIPE, cwd=repo_dir_path)

    return compile_proc.returncode


def run_test(source, repo_dir_path):
    bugg = False
    compile_fail = False
    timed_out = False
    entire_bugg = False
    log = ""
    failing_tests = 0

    try:
        tokens = javalang.tokenizer.tokenize(source)
        parser = javalang.parser.Parser(tokens)
        parser.parse()
    except:
        print("Syntax Error")
        return compile_fail, timed_out, bugg, entire_bugg, True, None

    if compile_repo(repo_dir_path) != 0:
        return True, timed_out, bugg, entire_bugg, True, None

    print('Check if it passes all the test')

    try:
        test_process = subprocess.run(['mvn', 'test', '-Drat.skip=true'], capture_output=True, cwd=repo_dir_path, timeout=90)
        captured_stdout = test_process.stdout.decode()
        print(captured_stdout)
        pattern = re.compile(r'Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)')
        matches = pattern.findall(captured_stdout)
        failures = int(matches[-1][1])
        errors = int(matches[-1][2])
        failing_tests = failures + errors

    except subprocess.TimeoutExpired:
        timed_out = True
    except Exception as e:
        return True, timed_out, bugg, entire_bugg, True, None

    if not timed_out and failing_tests == 0:
        print('success')
    else:
        entire_bugg = True

    return compile_fail, timed_out, bugg, entire_bugg, False, log


def validate_patch_rwb(file, dataset="RWBV1.0"):
    if dataset == "RWBV1.0":
        with open("../Datasets/RWB/RWB-V1.0.json", "r") as f:
            bug_dict = json.load(f)
    else:
        with open("./Datasets/RWB/RWB-V2.0.json", "r") as f:
            bug_dict = json.load(f)

    current_file = file.split('/')[-1]
    bug_id = current_file.split('.')[0]
    project = bug_id.split("-")[0]
    bug = bug_id.split("-")[1]
    start = bug_dict[bug_id]['start']
    end = bug_dict[bug_id]['end']
    fixed_class_path = bug_dict[bug_id]["fixed_class_path"]

    print(current_file, bug_id)

    git_reset(f"../Datasets/RWB/{project}/{bug}f")
    git_clean(f"../Datasets/RWB/{project}/{bug}f")

    with open(file, 'r') as f:
        patch = f.readlines()
    
    try:
        with open(f"../Datasets/RWB/{project}/{bug}f/{fixed_class_path}", 'r') as f:
            source = f.readlines()
    except:
        with open(f"../Datasets/RWB/{project}/{bug}f/{fixed_class_path}", encoding='ISO-8859-1') as f:
            source = f.readlines()
    
    source = "\n".join(source[:start - 1] + patch + source[end:])

    try:
        with open(f"../Datasets/RWB/{project}/{bug}f/{fixed_class_path}", 'w') as f:
            f.write(source)
    except:
        with open(f"../Datasets/RWB/{project}/{bug}f/{fixed_class_path}", 'w', encoding='ISO-8859-1') as f:
            f.write(source)

    compile_fail, timed_out, bugg, entire_bugg, syntax_error, log = run_test(source, f"../Datasets/RWB/{project}/{bug}f")

    if not compile_fail and not timed_out and not bugg and not entire_bugg and not syntax_error:
        print("{} has valid patch: {}".format(bug_id, file))
        return True, None
    else:
        compile_fail, timed_out, bugg, entire_bugg, syntax_error, log
        print("{} has invalid patch: {}".format(bug_id, file))
        if compile_fail: message = "Compile Fail"
        elif timed_out: message = "Time Out"
        elif syntax_error: message = "Syntex Error"
        else: message = "Test Fail"
        return False, message