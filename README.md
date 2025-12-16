# Faico: Faithful and Complete Knowledge Graph Augmented Reasoning

This repository contains the implementation of **Faico**, a Knowledge Graph (KG) augmented reasoning framework designed to achieve both **semantic faithfulness** and **structural completeness**.

**Faico** decouples model inference from graph traversal by integrating:

1. **Token-Trie Guided Relation Generator:** A fine-tuned LLM that generates schema-valid and query-relevant relation types.

2. **k-BET Subgraph Retriever:** A budget dominance based graph algorithm that efficiently identifies the maximal $k$-bounded edge type subgraph.



If you prefer to skip training stage, we provide a  [quick-eval scripts](#quickstart) along with intermediate results so that you can easily reproduce/re-evaluate the results under qwen or different LLM backbone.

## 📂 Project Structure

```text
#You may need to download some data into some directories
Faico
├── data/                       # Dataset files and Knowledge Graph data
│   ├── dataset/                # CWQ, WebQSP, GrailQA json files
│   └── freebase/               # Freebase KG files
├── results/                    # Output directories for intermediate and eval results
│   ├── predicted_relation_res/ # Generated edge labels from Step 2
│   ├── k-BET_res/              # Retrieved k-BET subgraphs from Step 3
│   └── predict_answers_res/    # Final LLM answers from Step 4
├── src/                        # Source code
│   ├── finetune/               # Code for SFT and Token-Trie decoding
│   │   ├── sft_1GPU.py         # Step 1: Fine-tuning script
│   │   ├── RelTokenTrie.py     # Token-Trie construction
│   │   └── TreeDecode.py       # Step 2 Constrained decoding logic
│   ├── graph/                  # Graph algorithms and reasoning
│   │   ├── KnowledgeBase.py    # KG loading and operations
│   │   ├── k-BET.py            # Step 3: k-BET Subgraph Retrieval (Algorithm 1)
│   │   ├── generate_answers.py # Step 4: LLM Reasoning phase
│   │   ├── config.py           # Configuration parameters
│   │   ├── prompts.py          # LLM prompt templates
│   │   └── util.py             # Utility functions
│   └── eval/                   # Evaluation scripts
│       ├── eval.py             # Main evaluation script
│       └── graph_eval.py       # Graph-related evaluation
├── requirements.txt            # Python dependencies, full version
└── requirements_graph_eval.txt # Python dependencies for quick eval
```

## 🛠️ Installation & Preprocessing

### 1. Clone the repository:

   ```bash
   git clone https://github.com/leviGuoCheng/Faico.git
   cd Faico
   ```

### 2. Install dependencies:

   ```bash
   conda create -n faico python=3.11
   conda activate faico
   pip install -r requirements.txt
   
   # if you only want to run k-BET and eval scripts, you can just
   # pip install -r requirements_graph_eval.txt
   ```


### 3. Model & Data Preparation

We utilize **[DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)** as the base model. Follow the steps below to setup the model and data.

#### A. Download Model

```
# Download the model using huggingface
hf download deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --local-dir ./data/model/DeepSeek-R1-0528-Qwen3-8B

# optional : modelscope
# pip install modelscope
# modelscope download --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --local_dir ./data/model/DeepSeek-R1-0528-Qwen3-8B
```

#### B. Download & Extract Datasets

Download the 

1. Benchmark Datasets for training (WebQSP, CWQ, GrailQA)

Download the [Faico-train-data.zip](https://drive.google.com/drive/folders/1ZsDZ6NcRI24ML5B8S5Pn8eR9OxSwBwGi?usp=sharing) file and extract it into `data/dataset`.

2. Freebase Knowledge Graph

Download the [freebase.tar.gz](https://drive.google.com/drive/folders/1ZsDZ6NcRI24ML5B8S5Pn8eR9OxSwBwGi?usp=sharing) and extract it into `data/freebase`.

3. Pre-computed k-BET graph results (for direct evaluation)
   Download the [kBET_res.tar.gz](https://drive.google.com/file/d/1y7_dAUDzOK9EOxbVXWn2FC5RPSZRcNQ7/view?usp=share_link) and extract it into `results/k-BET_res`

```bash
#After placing the downloaded fils in Faico
# cd path_to_faico/Faico
unzip Faico-train-data.zip -d data/dataset
#you would see data/dataset/[cwq\webqsp\grailqa]/*[train\dev\test].json

tar -zxvf freebase.tar.gz -C data/
#you would see data/freebase/*.bin

tar -zxvf kBET_res.tar.gz -C results/
#you would see results/k-BET_res/[cwq\webqsp\grailqa]/k-BET_res.json

#optional clean-up
rm freebase.tar.gz
rm benchmarks.zip
rm kBET_res.tar.gz
```


(Or you can download them all from [Quark Drive](https://pan.quark.cn/s/6192131b216b),code: uq5D)

##  ⚙️ Complete Pipeline

The Faico pipeline consists of four sequential stages followed by evaluation.

### Step 1: Relation Type Learning (Fine-tuning)

we fine-tune a general-purpose LLM (e.g., Qwen) to predict valid relation types based on the KG schema. This utilizes `src/finetune/sft_1GPU.py`.

```bash
#If there is a dev set, add it using '--val_json', or we just use part of the train set as validation set.

# webqsp
CUDA_VISIBLE_DEVICES=0 python sft_1GPU.py \
--model_dir   ./data/model/DeepSeek-R1-0528-Qwen3-8B \
--train_json  ./data/dataset/webqsp/WebQSP.train.json \
--output_dir  ./data/model/finetune/webqsp-dpsk-R1-distall-Qwen3-8B-test \
--num_samples -1 \
--batch_size 32 \
--epochs 3 \
--lr 2e-5 \
--r 64 \
--lora_alpha 128 \
--lora_dropout 0.05 \
--no_mask

```

### Step 2: Relation Label Generation

Use the fine-tuned model to generate query-relevant relation types ($\mathcal{R}_q$). This step uses the **Token-Trie** constraint to ensure all generated relations exist in the KG schema.

*Output location:* `results/predicted_relation_res/`

```bash
# Generate relation labels for the test set
CUDA_VISIBLE_DEVICES=0 python3 TreeDecode.py \
--model_path /home/chengguo/data/models/finetune/webqsp-dpsk-R1-distill-Qwen3-8B-test \
--test_data ./data/datasets/webqsp/WebQSP.test.json \
--bin_file ./data/freebase/edge_label_to_id.bin \
--save_file webqsp_relation_label \
--num_samples -1
```

### Step 3: k-BET Subgraph Retrieval

Based on the generated relation types, retrieve the **maximal k-Bounded Edge Type (k-BET)** subgraph. This uses the DFS-based algorithm with budget dominance pruning implemented in `src/graph/k-BET.py`.

*Default Output location:* `results/k-BET_res`


❗ **Critical Setup Step**  
You **must** set the parameters in `src/graph/config.py` before running k-BET retrieval, answer generation, or any downstream evaluation.

We have prepared relation-label-generation results for every dataset; you can use them directly from `results/predicted_relation_res`.

We have prepared ready-to-use k-BET retrieval results for every dataset; simply download them from [here](https://drive.google.com/file/d/1y7_dAUDzOK9EOxbVXWn2FC5RPSZRcNQ7/view?usp=share_link). These pre-computed subgraphs can be directly fed into the downstream answer-generation stage or used as input for graph-level evaluation.

```bash
# Run k-BET subgraph retrieval
#cwq
python src/graph/k-BET.py \
--input  ./results/predicted_relation_res/cwq_relation_label.json \
--dataset cwq \
--depth 4 \
--k 1 \
--description "CWQ dataset with 4-depth exploration" \
--save_dir ./results/k-BET_res/cwq_4depth 

#webqsp
python src/graph/k-BET.py \
--input  ./results/predicted_relation_res/webqsp_relation_label.json \
--dataset webqsp \
--depth 2 \
--k 1 \
--description "WebQSP dataset with 2-depth exploration" \
--save_dir ./results/k-BET_res/webqsp_2depth 

#grailqa
python src/graph/k-BET.py \
--input  ./results/predicted_relation_res/grailqa_relation_label.json \
--dataset grailqa \
--depth 4 \
--k 1 \
--description "GrailQA dataset with 4-depth exploration" \
--save_dir ./results/k-BET_res/grailqa_4depth 

```

<a id="generate"></a>

### Step 4: LLM Answer Generation

Feed the original question and the retrieved k-BET subgraph (serialized as triples) into the LLM to generate the final natural-language answer. The corresponding code is located at `src/graph/generate_answers.py`. 

*Output location:* `results/predict_answers_res/`

The generated results will be saved as a JSON file, which you can use for subsequent evaluation.

```bash
# Generate final answers using the retrieved subgraphs
# For CWQ dataset
python src/graph/generate_answers.py \
--dataset cwq \
--model qwen-plus \
--num_threads 3 \
--input results/k-BET_res/cwq/k-BET_result.json \
--output cwq_answers

# For WebQSP dataset
python src/graph/generate_answers.py \
--dataset webqsp \
--model qwen-plus \
--num_threads 3 \
--input results/k-BET_res/webqsp/k-BET_result.json \
--output webqsp_answers

# For GrailQA dataset
python src/graph/generate_answers.py \
--dataset grailqa \
--model qwen-plus \
--num_threads 3 \
--input results/k-BET_res/grailqa/k-BET_result.json \
--output grailqa_answers
```

<a id="eval"></a>

### Step 5: Evaluation

Evaluate the performance of **Faico** using standard metrics (Hit@1, Precision, Recall, F1).

```bash
# Run evaluation script
python src/eval/eval.py \
--dataset cwq \
--input results/predict_answers_res/cwq_answers.json 
```

To evaluate the quality of our retrieved subgraphs, please use `src/eval/graph_eval.py`.

```bash
# Run subgraph evaluation script
python src/eval/graph_eval.py \
--dataset cwq \
--input results/k-BET_res/cwq/k-BET_result.json
```



<a id="quickstart"></a>

## 🚀 QuickStart

Here we offer a quickstart to reproduce **Faico** results with intermediate graph data [kBET_res](https://drive.google.com/file/d/1y7_dAUDzOK9EOxbVXWn2FC5RPSZRcNQ7/view?usp=share_link).

```bash
conda create -n faico_eval python=3.11
conda activate faico_eval
pip install -r requirements_graph_eval.txt

#Download the kBET_res.tar.gz and put it into Faico/
tar -zxvf kBET_res.tar.gz -C results/

#Configure your keys in src/graph/config.py
```

Run [step4](#generate) and [step 5](#eval).

If you want to swap a different LLM backbone, modify the **BASE_URL** (and related  **API_KEY**) in `src/graph/config.py`, set a different ` –-model` arguement when running `src/graph/generate_answers.py`.

## 📊Metrics & Performance

The framework evaluates the results with complete answer set ( taking multiple answers into account with more emphasis on set-based exact match) performance based on:

  * **Answer Quality:** Hit*, Hit@1, Precision, Recall, F1.
  * **Subgraph Quality:** Edge Recall, Edge Precision, Answer Coverage , etc.

Please see the Experiments section in our paper for details.
