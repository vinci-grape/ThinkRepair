# ThinkRepair: Self-Directed Automated Program Repair

This repo contains both the correct patches generated by our study along with the code used to run the experiment.

### Prerequisites

```
Install Defects4J from https://github.com/rjust/defects4j 
export PATH=$PATH:"path2defects4j"/framework/bin
```

```
JDK 1.8 for Defects4J
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
```

```
Download RWB datasets from https://figshare.com/s/b8fa6587bfcb9908b0da
```

### Plausible patches generation

Example usage to run ThinkRepair_ChatGPT:
select: Selection strategy to use, current support: CSelect, SSelect, RSelect.
pfl: Use the perfect fault information.
n_example: Number of examples.
sample: The maximum number of repair attempts.
chance: The maximum number of interactions.
api_key: Place your ChatGPT access key.
```
python repair_chatgpt.py --dataset D4JV1.2 \
               --select CSelect \
               --pfl True \
               --n_example 2 \
               --sample 25 \
               --chance 5
```

Example usage to run ThinkRepair_open_source_llm:
STARCODER_AUTH_TOKEN: Place your starcoder auto token.
```
python repair_open_source.py --dataset D4JV1.2 \
               --select CSelect \
               --pfl True \
               --n_example 2 \
               --sample 25 \
               --chance 5 \
```

After running ThinkRepair, your directory structure should be like the following:

```
ThinkRepair
├── Datasets
    ├── D4J
        ├── generate_prompt*: input for the collection phase
        ├── repair_prompt*: input for the fixing phase
    ├── ...
├── Results
    ├── ChatGPT
        ├── D4JV1.2
            ├── cot_pfl.json
            ├── repair_pfl.json
        ├── ...
    ├── CodeLlama
        ├── D4JV1.2
            ├── cot_pfl.json
            ├── repair_pfl.json
        ├── ...
    ├── ...
├── ...
```