import json
import argparse
import os
import sys
import re

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.graph.KnowledgeBase import KnowledgeBase
from src.graph.config import FREEBASE_DIR

def extract_gt_relation(sparql_string,dataset):
    """
    sparql_string: String representation of sparql
    """     
    if dataset == "grailqa":
        pattern = r":([\w\.]+)\s"
    else:
        pattern = r"ns:(\S+)"
    matches = re.findall(pattern, sparql_string)
    matched_entity =  [item_ for item_ in matches if item_.count('.')<=1]
    matched_relation = [item_ for item_ in matches if item_ not in matched_entity ]

    return list(set(matched_relation))
def create_index(total_processed_datasets):
    """
    Create hash table indices for quick lookup by question and ID.
    Time complexity: O(n) for construction, O(1) for lookup.
    
    Args:
        total_processed_datasets: List of datasets to index
        
    Returns:
        tuple: (question_index, id_index) - Two dictionaries for quick lookup
    """
    question_index = {}
    id_index = {}
    
    for item in total_processed_datasets:
        # Index by question
        question = item.get("question")
        if question:
            question_index[question] = item
        
        # Index by ID
        item_id = item.get("ID")
        if item_id:
            id_index[item_id] = item
    
    return question_index, id_index

def find_the_item_in_total(data, total_processed_datasets, question_index=None, id_index=None):
    """
    Optimized function to find item by question or ID using hash table indices.
    
    Args:
        data: Dictionary with 'question' and/or 'id' keys to search for
        total_processed_datasets: List of datasets (only used if indices not provided)
        question_index: Optional pre-created index for question lookup
        id_index: Optional pre-created index for ID lookup
        
    Returns:
        The matching item if found
        
    Raises:
        ValueError: If no matching item is found
    """
    # Create indices if not provided
    if question_index is None or id_index is None:
        question_index, id_index = create_index(total_processed_datasets)
    
    # First try to find by ID (more reliable)
    data_id = data.get("id")
    if data_id and data_id in id_index:
        return id_index[data_id]
    
    # Then try to find by question
    question = data.get("question")
    if question and question in question_index:
        return question_index[question]
    
    # If no match found
    raise ValueError(f"Item not found for question: {question} or ID: {data_id}")
    


def extrat_answers(un_processed_answers_ls,dataset_name):
    """
    un_processed_answers_ls: List of unprocessed answers, which is a list of dicts
    """

    processed_answers_ls = []
    if dataset_name == "webqsp":
        for answer in un_processed_answers_ls:
            if answer["AnswerType"] == "Value":
                processed_answers_ls.append(answer["AnswerArgument"].lower())
            elif answer["AnswerType"] == "Entity":
                processed_answers_ls.append(answer["EntityName"].lower())
        return processed_answers_ls
    elif dataset_name == "grailqa":
        for answer in un_processed_answers_ls:
            if answer["answer_type"] == "Entity":
                processed_answers_ls.append(answer["entity_name"].lower())
            elif answer["answer_type"] == "Value":
                processed_answers_ls.append(answer["answer_argument"].lower())
        return processed_answers_ls

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--dataset", type=str, required=True)
    argparse.add_argument("--input", type=str, required=True)
    args = argparse.parse_args()

    if args.input.endswith(".jsonl"):
        with open(args.input, "r") as f:
            input_data_items = f.readlines()
            input_data_items = [json.loads(data) for data in input_data_items]
    else:
        # Indicates a json file
        with open(args.input, "r") as f:
            input_data_items = json.load(f)


# ===================================Prepare Dataset==================================================================
# ===========================================================================================================
    if args.dataset == "cwq":
        total_processed_datasets = []
        id_string = "ID"
        question_string = "question"
        answer_string = "answer" #List, all human-readable content
        sparql_string = "sparql"
        with open(project_root + "/data/dataset/cwq.json","r") as file:
            datasets = json.load(file)
        for data in datasets:
            processed_data = {}
            processed_data["ID"] = data[id_string]
            processed_data["question"] = data[question_string]
            info = []
            info.append({"answers":[ans.lower() for ans in data[answer_string]],"gt_relations":extract_gt_relation(data[sparql_string],args.dataset)})
            processed_data["info"] = info
            processed_data["topic_entity"] = list(data['topic_entity'].values())
            processed_data["answers"] = [ans.lower() for ans in data[answer_string]]
            total_processed_datasets.append(processed_data)
    elif args.dataset =="webqsp":
        total_processed_datasets = []
        with open(project_root + "/data/dataset/WebQSP.json", "r") as f:
            datasets = json.load(f)
            id_string = "QuestionId"
            question_string = "RawQuestion"
            answer_string = "Answers"
            sparql_string = "Sparql"
        for data in datasets:
            processed_data = {}
            processed_data["ID"] = data[id_string]
            processed_data["question"] = data[question_string]
            info = []
            for parse in data["Parses"]:
                gt_relations_in_parse = extract_gt_relation(parse["Sparql"],args.dataset)
                # 
                answers_in_parse = extrat_answers(parse["Answers"],args.dataset)
                info.append({"answers":answers_in_parse,"gt_relations":gt_relations_in_parse})

            processed_data["info"] = info
            processed_data["topic_entity"] = list(data['topic_entity'].values())
            answers = data["Parses"]
            answer_ls = []
            for answer in answers:
                for name in answer['Answers']:
                    if name['EntityName'] == None:
                        answer_ls.append(name['AnswerArgument'])
                    else:
                        answer_ls.append(name['EntityName'])
            processed_data["answers"] = list(set(answer_ls))

            total_processed_datasets.append(processed_data)
    elif args.dataset =="grailqa":
        total_processed_datasets = []
        with open(project_root + "/data/dataset/grailqa.json", "r") as f:
            datasets = json.load(f)
            id_string = "qid"
            question_string = "question"
            answer_string = "answer" #This is a list, each item is a dict
            sparql_string = "sparql_query"
        for data in datasets:
            processed_data = {}
            processed_data["ID"] = data[id_string]
            processed_data["question"] = data[question_string]
            info = []
            # 
            info.append({"answers":extrat_answers(data[answer_string],args.dataset),"gt_relations":extract_gt_relation(data[sparql_string],args.dataset)})
            processed_data["info"] = info
            processed_data["data"] = data
            total_processed_datasets.append(processed_data)
    

# ===========================================Start Evaluation=================================================================
# ===============================================================================================================
    # total_processed_datasets contains the dataset content
    # Start processing each baseline below
    KB = KnowledgeBase(FREEBASE_DIR)
    # This requires a lot of processing, getting the k-BET results

    inputs = []
    for key , input_item in input_data_items.items():
        big_graph = input_item["pre_triplets"]
        small_graph = input_item["triplets"]

        if len(big_graph)<=1000:
            graph = big_graph
        else:
            graph = small_graph
        
        reasoning_chains = []
        for triples in graph:
            reasoning_chains.append([KB.get_entity_name(triples[0]),triples[1],KB.get_entity_name(triples[2])])
        inputs.append({"id":input_item[id_string],"question":input_item[question_string],"reasoning_chains":reasoning_chains})
    input_data_items = inputs

    # Create indices once for efficient lookup
    question_index, id_index = create_index(total_processed_datasets)

    triples_cnt = 0
    total_gt_relation_recall = 0
    total_gt_relation_precision = 0
    total_answer_recall = 0
    total_gt_hit = 0
    total_answer_hit = 0
    total_answer_hit_1 = 0
    for input_item in input_data_items:
        triplets_ls = input_item["reasoning_chains"]
        triplets_ls = [[item.lower() for item in triple] for triple in triplets_ls]
        triples_cnt += len(triplets_ls)

        dataset_item = find_the_item_in_total(input_item, total_processed_datasets, question_index, id_index)#Try to find that item

        # There may be multiple info items
        best_gt_relation_recall = -1
        best_gt_relation_precision = -1
        best_answer_recall = -1
        best_gt_hit = -1
        best_answer_hit = -1
        best_answer_hit_1 = -1
        for answer_info in dataset_item['info']:
            used_relation = [i_[1] for i_ in triplets_ls if i_ != []]
            gt_relations = answer_info['gt_relations']
            gt_relations = [i_ for i_ in gt_relations if i_ != "type.object.type"]
            used_nodes = [i_[0] for i_ in triplets_ls if i_ != []] + [i_[2] for i_ in triplets_ls if i_ != []]
            gt_nodes = answer_info['answers']

            # Start this calculation
            gt_relation_recall = len(set(used_relation) & set(gt_relations)) / len(set(gt_relations)) #No problem
            # The gt_precision here should calculate the proportion of elements in used_relation list that are also in gt_relations
            
            gt_relation_precision = len([r for r in used_relation if r in gt_relations]) / len(used_relation) if len(used_relation) != 0 else 0
            answer_recall = len(set(used_nodes) & set(gt_nodes)) / len(set(gt_nodes)) if len(set(gt_nodes)) != 0 else 1 #No problem
            
            # If all gt_relations are contained in used_relation, then gt_hit is 1, otherwise 0
            gt_hit = 1 if set(gt_relations).issubset(set(used_relation)) else 0
            # If all gt_nodes are contained in used_nodes, then answer_hit is 1, otherwise 0
            answer_hit = 1 if set(gt_nodes).issubset(set(used_nodes)) else 0
            answer_hit_1= 0
            for node in gt_nodes:
                if node in used_nodes:
                    answer_hit_1 = 1
                    break
            
            if gt_relation_recall > best_gt_relation_recall:
                best_gt_relation_recall = gt_relation_recall
            if gt_relation_precision > best_gt_relation_precision:
                best_gt_relation_precision = gt_relation_precision
            if answer_recall > best_answer_recall:
                best_answer_recall = answer_recall
            if gt_hit > best_gt_hit:
                best_gt_hit = gt_hit
            if answer_hit > best_answer_hit:
                best_answer_hit = answer_hit
            if answer_hit_1 > best_answer_hit_1:
                best_answer_hit_1 = answer_hit_1
        total_gt_relation_recall += best_gt_relation_recall
        total_gt_relation_precision += best_gt_relation_precision
        total_answer_recall += best_answer_recall
        total_gt_hit += best_gt_hit
        total_answer_hit += best_answer_hit
        total_answer_hit_1 += best_answer_hit_1


    # Calculate the average of the metrics
    avg_gt_relation_recall = total_gt_relation_recall / len(input_data_items)
    avg_gt_relation_precision = total_gt_relation_precision / len(input_data_items)
    avg_answer_recall = total_answer_recall / len(input_data_items)
    avg_total_triples = triples_cnt / len(input_data_items)
    avg_gt_hit = total_gt_hit / len(input_data_items)
    avg_answer_hit = total_answer_hit / len(input_data_items)
    avg_answer_hit_1 = total_answer_hit_1 / len(input_data_items)
    
    # Print evaluation results in a clear format
    print("\n" + "="*60)
    print("Graph Evaluation Results")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Number of samples processed: {len(input_data_items)}")
    print("="*60)
    print(f"Ground Truth Relation Recall:    {avg_gt_relation_recall:.3f} (Recall of GT relations in retrieved graph)")
    print(f"Ground Truth Relation Precision: {avg_gt_relation_precision:.3f} (Precision of retrieved relations w.r.t GT)")
    print(f"Answer Recall:                   {avg_answer_recall:.3f} (Recall of GT answers in retrieved graph)")
    print(f"GT Relation Hit:                 {avg_gt_hit:.3f} (Proportion of samples with all GT relations hit)")
    print(f"Answer Hit:                      {avg_answer_hit:.3f} (Proportion of samples with all GT answers hit)")
    print(f"At least one Answer Hit:         {avg_answer_hit_1:.3f} (Proportion of samples with at least one GT answer hit)")
    print(f"Average Triples in Graph:        {avg_total_triples:.2f} (Average number of triples in retrieved graph)")
    print("="*60)



            
        

    
    
