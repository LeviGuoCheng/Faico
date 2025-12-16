import argparse
import logging
import time
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import json, math, re
from RelTokenTrie import create_token_trie
import random
import ast
from tokenizers import Tokenizer
import os

def get_train_data_(file_path, tokenizer_path, num_samples=-1,is_mask:bool=False):
    """
    Generate training data from a file containing SPARQL queries and questions.
    
    Args:
        file_path: Path to input JSON file with SPARQL queries
        tokenizer_path: Path to the tokenizer model
        num_samples: Number of samples to process, -1 for all samples
        
    Returns:
        List of dictionaries containing training examples
    """
    with open(file_path, "r") as file:
        if(file_path.endswith('.json')):
            data = json.load(file)
        elif(file_path.endswith('.jsonl')):
            data = [json.loads(line) for line in file]
    
    train_data = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # json_file = os.path.join(tokenizer_path, "tokenizer.json")
    # tokenizer = Tokenizer.from_file(json_file)
    
    
    if('cwq' in file_path or 'complex' in file_path.lower()):
        process_data = data[0:num_samples] if num_samples != -1 else data
        question_key_str = "question" 
        question_raw_str = "machine_question"
        for item in process_data:
            if item.get("sparql") is not None:
                # start processing
                sqarql_query = item["sparql"]
                if "augmented_queries" in item:
                    query_list = item["augmented_queries"]
                else:
                    query_list = [item[question_raw_str]]
                query_original = item[question_key_str]
                query_list.append(query_original)
                pattern = r"ns:(\S+)"
                matches = re.findall(pattern, sqarql_query)
                #matched_entity = [item_ for item_ in matches if item_[1]=='.' and item_.count('.')==1]
                matched_entity = [item_ for item_ in matches if item_.count('.')<=1]
                matched_relation = [item_ for item_ in matches if (item_ not in matched_entity) and (item_.startswith("common.")==False)]
                trie = create_token_trie(matched_relation, tokenizer)
                for query in query_list:
                    train_data.extend(trie.get_train_data(query,is_mask))
    elif('webqsp' in file_path.lower()):
        question_raw_str = "RawQuestion" 
        question_processed_str = "ProcessedQuestion"
        process_data = data["Questions"][0:num_samples] if num_samples != -1 else data["Questions"]
        for item in process_data:
            if item.get("Parses") is not None:

                ##Get question list
                parses = item["Parses"]
                question_original = item[question_raw_str]
                question_processed = item[question_processed_str]
                question_list = [question_original]
                # for most question, the processed question is the same as the original question, only differ in a question mark
                # so we skip the identical question
                if question_processed is not None and question_processed.rstrip('?') != question_original.rstrip('?'):
                    question_list.append(question_processed)
                ## For webqsp, there are multiple parses for each question, we need to get all possible relation labels
                ## Get all possible relation labels
                for parse in parses:
                    if parse.get("Sparql") is not None:
                        sparql_query = parse["Sparql"]
                        pattern = r"ns:(\S+)"
                        matches = re.findall(pattern, sparql_query)
                        #matched_entity = [item_ for item_ in matches if item_[1]=='.' and item_.count('.')==1]
                        matched_entity = [item_ for item_ in matches if item_.count('.')<=1]
                        matched_relation = [item_ for item_ in matches if (item_ not in matched_entity) and (item_.startswith("common.")==False)]
                trie = create_token_trie(matched_relation, tokenizer)
                for query in question_list:
                    train_data.extend(trie.get_train_data(query,is_mask))
    elif('grailqa' in file_path.lower()):
        process_data = data[0:num_samples] if num_samples != -1 else data
        for item in process_data:
            if item.get("sparql_query") is not None:
                sparql_query = item["sparql_query"]
                query = item["question"]
                pattern = r":([\w\.]+)\s"
                matches = re.findall(pattern, sparql_query)
                matched_entity = [item_ for item_ in matches if item_.count('.')<=1] # :m.*****, skip
                matched_relation = [item_ for item_ in matches if (item_ not in matched_entity) and (item_.startswith("common.")==False)] # evict : and the last space
                trie = create_token_trie(matched_relation, tokenizer)
                train_data.extend(trie.get_train_data(query,is_mask))
    else:
        raise ValueError(f"Unsupported dataset: {file_path}")
    return train_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',   type=str, required=True)
    parser.add_argument('--train_json',  type=str, required=True)
    parser.add_argument('--val_json',    type=str, default=None)
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--epochs',      type=int, default=3)
    parser.add_argument('--lr',          type=float, default=2e-5)
    parser.add_argument('--r',           type=int, default=64)
    parser.add_argument('--lora_alpha',  type=int, default=128)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mask', dest='is_mask', action='store_true', help='mask on')
    group.add_argument('--no_mask', dest='is_mask', action='store_false', help='mask off')
    parser.set_defaults(is_mask=False)

    return parser.parse_args()

def tokenize_fn(ex, tokenizer):
    out = tokenizer(ex['text'], truncation=True, max_length=512)
    # convert target token strings to ids into parallel lists
    ids, probs = [], []
    for tok_str, p in ex['target_probs'].items():
        if p == None:
            continue
        tok_id = tokenizer.convert_tokens_to_ids(tok_str)
        if tok_id == tokenizer.unk_token_id:
            logging.error(f"Unrecognized label token: {tok_str}")
            print(ex['text'])
            continue
        ids.append(tok_id)
        probs.append(p)
    out['target_ids'] = ids      # List[int]
    out['target_probs'] = probs  # List[float]
    return out

def collate_fn(features, tokenizer,is_mask:bool=False):
    batch = tokenizer.pad(
        {
            "input_ids":      [f["input_ids"]      for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        },
        return_tensors="pt"
    )
    masked_positions = []
    all_ids, all_probs = [], []
    for f in features:
        all_ids.append(f["target_ids"])
        all_probs.append(f["target_probs"])
    for item in batch["input_ids"]:
        if is_mask:
            pos = (torch.tensor(item) == tokenizer.mask_token_id).nonzero()
            assert pos.numel() == 1, "Only one masked token per sample!"
            masked_positions.append(pos[0].item())
        else:
            pos = (torch.tensor(item) == tokenizer.eos_token_id).nonzero()
            if pos.numel() == 0:
                pos = torch.tensor([[len(item)]])
            masked_positions.append(pos[0].item()-1)
    batch["masked_positions"] = torch.tensor(masked_positions, dtype=torch.long)
    batch["target_ids"]       = all_ids
    batch["target_probs"]     = all_probs
    return batch
def compute_loss(model, inputs, return_outputs=False):
    id_lists   = inputs.pop("target_ids")    # List[List[int]]
    prob_lists = inputs.pop("target_probs")  # List[List[float]]
    poss       = inputs.pop("masked_positions")

    out  = model(**inputs)
    logp = torch.log_softmax(out.logits, dim=-1)  # [B, L, V]
    losses = []
    for i, pos in enumerate(poss):
        lp = logp[i, pos, :]                      # [V]
        # construct target distribution
        target = lp.new_zeros(lp.shape)
        for tok_id, p in zip(id_lists[i], prob_lists[i]):
            target[tok_id] = p
        # KL(p||q) = Σ p * (log p - log q)
        losses.append(torch.sum(target * (torch.log(target + 1e-20) - lp)))
    loss = torch.stack(losses).mean()
    return (loss, out) if return_outputs else loss


class MySFTTrainer(SFTTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        return compute_loss(model, inputs, return_outputs)


def main():
    args = parse_args()
    if args.is_mask:
        args.output_dir = args.output_dir + "_mask"
    logging.basicConfig(level=logging.INFO)

    # —— tokenizer & mask_token —— 
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print(tok.is_fast)
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    if args.is_mask:
        tok.mask_token = "<mask>"
        if '<mask>' not in tok.get_vocab():
            tok.add_special_tokens({'additional_special_tokens': ['<mask>']})
    # model loading 
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True
    ).to('cuda').half()
    model.resize_token_embeddings(len(tok))

    # dataset construction
    load_train_data_start_time = time.time()
    train_list = get_train_data_(args.train_json, args.model_dir, args.num_samples,is_mask=args.is_mask)
    load_train_data_end_time = time.time()
    print(f"Loading train data time: {load_train_data_end_time - load_train_data_start_time} seconds")
    print(f"Using {len(train_list)} samples from train set")
    if args.val_json is not None:
        val_list   = get_train_data_(args.val_json,   args.model_dir, args.num_samples,is_mask=args.is_mask)
    else:
        # webqsp has no dev set
        val_list = random.sample(train_list, int(0.1*len(train_list)))
        print(f"Using {len(val_list)} samples from train set as validation set")
    print(f"loading validation data time: {time.time() - load_train_data_end_time} seconds")
    tokenize_train_data_start_time = time.time()
    train_ds = Dataset.from_list(train_list).map(lambda ex: tokenize_fn(ex, tok))
    tokenize_train_data_end_time = time.time()
    print(f"Tokenizing train data time: {tokenize_train_data_end_time - tokenize_train_data_start_time} seconds")


    tokenize_val_data_start_time = time.time()
    val_ds   = Dataset.from_list(val_list).map(lambda ex: tokenize_fn(ex, tok))
    tokenize_val_data_end_time = time.time()
    print(f"Tokenizing val data time: {tokenize_val_data_end_time - tokenize_val_data_start_time} seconds")


    # Check the batch :). 
    from torch.utils.data import DataLoader
    dl = DataLoader(train_ds,
                    batch_size=16,
                    collate_fn=lambda f: collate_fn(f, tok))
    batch = next(iter(dl))
    print("DEBUG BATCH KEYS:", batch.keys())
    print("input_ids shape:", batch['input_ids'].shape)
    # If we get KeyError here, there maybe something wrong in tokenization or mapping

    #Training has started 
    hf_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        remove_unused_columns=False,
        bf16=True,
        logging_steps=5,
        save_strategy='no',
        eval_strategy='no',
        report_to='none'
    )
    sft_cfg = SFTConfig(**hf_args.to_dict(),
                        dataset_kwargs={'skip_prepare_dataset': True})
    peft_cfg = LoraConfig(
        r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj'],
        bias='none', task_type='CAUSAL_LM'
    )
    trainer = MySFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda f: collate_fn(f, tok,args.is_mask),
        #compute_loss_func=compute_loss,
        peft_config=peft_cfg
    )

    trainer.train()
    trainer.save_model()
    tok.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
