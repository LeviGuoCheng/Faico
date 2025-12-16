import json
import threading
import numpy as np
import os
import argparse
from KnowledgeBase import KnowledgeBase
from tqdm import tqdm
from config import FREEBASE_DIR
import time
import re
from util import perform_reasoning,check_string,clean_results


def generate_answer_from_paths(value,KB,model=None,key="triplets",temperature=0.0,max_tokens=2560):

    chains = value[key]
    
    chains_list = []
    label_id_map = {}
    for triples in chains:
        if triples[0].startswith("m."):
            entity_label0 = KB.get_entity_name(triples[0])
        else:
            entity_label0 = triples[0]
        if triples[2].startswith("m."):
            entity_label1 = KB.get_entity_name(triples[2])
        else:
            entity_label1 = triples[2]
        chains_list.append([triples[0],entity_label0,triples[1],triples[2],entity_label1])
        if entity_label0 not in label_id_map:
            label_id_map[entity_label0] = triples[0]   
        if entity_label1 not in label_id_map:
            label_id_map[entity_label1] = triples[2]
    
    prompt_chain = [(item[1],item[2],item[4]) for item in chains_list]

    results = perform_reasoning(value["question"],prompt_chain,model,temperature,max_tokens)
    print("llm response: ",results)

            
    if check_string(results):
        response = clean_results(results)
        if response=="NULL":
            response = results
    else:
        response = results
    print("[predict_answers]:",response)
    value["pre_answer"] = [response]
    return value
    

def process_chunk(chunk, thread_index, result_dict,model,KB,temperature,max_tokens):
    local_result = {}

    for key, value in tqdm(chunk,desc=f"Thread {thread_index}"):
        
        print("====================================================================================")
        print(f"question ID:  {value['ID']}")

        print(f"question:  {value['question']}")
        start_time = time.time()

        if len(value["pre_triplets"])<=1000:
            value = generate_answer_from_paths(value,KB,model=model,key="pre_triplets",temperature=temperature,max_tokens=max_tokens)
        else:
            value = generate_answer_from_paths(value,KB,model=model,temperature=temperature,max_tokens=max_tokens)
            
    
        
        predict_answer_lst = value["pre_answer"]
        try:
            if len(predict_answer_lst) == 1 and "," in predict_answer_lst[0]:
                predict_answer_lst = predict_answer_lst[0].split(',')
                cleaned_lst = [s.strip() for s in predict_answer_lst]
                predict_answer_lst = cleaned_lst
                cleaned_lst = [re.sub(r'^[^a-zA-Z0-9,]*', '', re.sub(r'[^a-zA-Z0-9,]*$', '', s.strip())) for s in predict_answer_lst]
                predict_answer_lst = cleaned_lst
        except Exception as e:
            print(f"[ERROR] processing answer_ls error :{e}")

        value["predict_answers"] = predict_answer_lst
        if "pre_answer" in value:
            del value["pre_answer"]
        
        if "time" in value:
            value["time"] = value["time"] + (time.time() - start_time)
        local_result[value["ID"]] = value
        
    result_dict[thread_index] = local_result




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='using llm to answer question')
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='Input JSON file path which is the result from lcr phase')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output text file name without file name extension ')
    parser.add_argument('-m', '--model', type=str, default="qwen-plus",
                        help='LLM model')
    parser.add_argument('--test_num', type=int, default=-1,
                        help='The question number you want to test, -1 represente all questions')
    parser.add_argument('--dataset', type=str, default="cwq",
                        help='The dataset you want to test,cwq or webqsp or grailqa are supported')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='The temperature you want to use for llm call')
    parser.add_argument('--max_tokens', type=int, default=2560,
                        help='The max_tokens you want to use for llm call')
    parser.add_argument('--num_threads', type=int, default=20,
                        help='The number of threads you want to use, you should consider the RPM and TPM of llm call')
    args = parser.parse_args()

    num_threads = args.num_threads
 
    
    KB = KnowledgeBase(FREEBASE_DIR) 

    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    predict_answers_res_dir = os.path.join(current_dir, "../../results/predict_answers_res/")
   
    if not os.path.exists(predict_answers_res_dir):
        os.makedirs(predict_answers_res_dir)

    output_file_json = predict_answers_res_dir+args.output+".json"

    with open(args.input, "r") as f:
            data = json.load(f)
    if args.test_num != -1:
        items = list(data.items())[:args.test_num]
    else:
        items = list(data.items())
    data = dict(items)
    chunks = np.array_split(items, num_threads)

    threads = []
    results_by_thread = {}

    # create threadings
    for i in range(num_threads):
        t = threading.Thread(target=process_chunk, args=(chunks[i], i, results_by_thread,args.model,KB,args.temperature,args.max_tokens))
        threads.append(t)
        t.start()

    # wait all threading to complete
    for t in threads:
        t.join()

    # merge all results
    data_to_write = {}
    for local in results_by_thread.values():
        data_to_write.update(local)

    print(f"Writing results to {output_file_json}...")
    with open(output_file_json, "w") as f_out:
        json.dump(data_to_write, f_out, indent=4)
    print(f"Finished writing results to {output_file_json}")
    
