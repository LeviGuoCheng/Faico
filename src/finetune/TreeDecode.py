from typing import List, Dict, Optional
import json
import torch
import torch.nn.functional as F
import logging
from RelTokenTrie import TokenTrie,TrieNode, create_token_trie
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
import argparse
import re
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TrieRouter:
    def __init__(self, trie: TokenTrie, model: torch.nn.Module, device=None):
        """
        Initialize the TrieRouter with a TokenTrie and embedding models.
        
        Args:
            trie: The TokenTrie to route through
            model_path_or_name: Path or name of the embedding model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        # Set device
        # if device is None:
        #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # else:
        #     self.device = torch.device(device)
        
        #logger.info(f"Initializing TrieRouter with model: {model_path_or_name} on device: {self.device}")
        self.trie = trie
        self.tokenizer = trie.tokenizer
        
        # # Initialize model
        self.model = model
        # Cache for embeddings
        self.embedding_cache = {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"TrieRouter initialized successfully")
    
    def make_path_string(self,path:List[TrieNode],is_infer:bool=False) -> str:
        cur_str = ""
        for node in path:
            if node.is_split_token():
                if node.is_end_of_token():
                    cur_str = cur_str + node.token+"·"
                else:
                    cur_str = cur_str + node.token
            elif node.is_non_split_token():
                cur_str = cur_str + node.token+"·"
            else:
                raise ValueError(f"Invalid token: {node.token}")
        if not is_infer and cur_str.endswith("·"):
            # tend to predict next token, skip the "·"
            cur_str = cur_str.replace("*","").replace("^","")
            return cur_str[:-1]
        else:
            cur_str = cur_str.replace("*","").replace("^","")
            return cur_str

    def _get_next_token_logits(self, question: str, pre_path: List[TrieNode] = None,is_mask:bool=False) -> torch.Tensor:
        """
        Get the logits for the next token prediction given a question and previous tokens.
        This is a simplified version - actual implementation would depend on your model's API.
        
        Args:
            question: The input question
            pre_path: Previous trie nodes in the current path
            
        Returns:
            Tensor of logits for next token prediction
        """
        # In a real implementation, you would:
        # 1. Encode the question
        # 2. If there are previous tokens in the path, include them in the context
        # 3. Get the model's prediction for the next token
        relation = self.make_path_string(pre_path,is_infer=True)
        #relation_prompt = f"""Generate relation for question -> {question}\nAnswer: {relation}<mask>"""
        mask_str = "<mask>" if is_mask else ""  
        relation_prompt = f"""Generate relation for question -> {question}\nAnswer: {relation}{mask_str}"""
        logger.info(f"Generate relation for question : {question} \nAnswer: {relation}{mask_str}")
        inputs = self.tokenizer(relation_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
            logits = outputs.logits[0, -1, :]  # Get logits for next token prediction
        
        return logits
    def _mask_logits_by_trie(self, logits: torch.Tensor, current_node: TrieNode) -> torch.Tensor:
        """
        Mask the logits tensor to only allow tokens that are in the current node's children.
        
        Args:
            logits: The original logits tensor from the model
            current_node: The current position in the trie
            
        Returns:
            A masked logits tensor
        """
        # Create a copy of the logits
        masked_logits = logits.clone()
        
        # Get all valid next tokens from the trie
        valid_tokens = []
        for token in current_node.children.keys():
            valid_tokens.append(token.replace("^","").replace("*",""))
        valid_tokens = set(valid_tokens)
        logger.debug(f"Valid next tokens: {valid_tokens}")
        
        # Convert tokens to token IDs
        valid_token_ids = set()
        for token in valid_tokens:
            first_token = self.tokenizer.tokenize(token)[0]
            token_id = self.tokenizer.convert_tokens_to_ids(first_token)
            valid_token_ids.add(token_id)
        
        # Create a boolean mask where False indicates valid tokens
        mask = torch.ones_like(masked_logits, dtype=torch.bool)
        for token_id in valid_token_ids:
            mask[token_id] = False
        
        # Apply the mask - set all invalid tokens to -inf
        masked_logits[mask] = float('-inf')
        
        return masked_logits
    
    def route_question(self, question: str, beam_width: int = 3, threshold: float = 0.1,is_mask:bool=False) -> List[str]:
        """
        Route through the trie using BFS with token logits.
        
        Args:
            question: The input question
            beam_width: Number of paths to explore at each step
            threshold: Minimum similarity score to consider a token
            
        Returns:
            List of relations relevant to the question
        """
        logger.info(f"Starting route_question with beam_width={beam_width}, threshold={threshold}")
        results = []
        # Add depth to track the current level in the trie
        queue = [(self.trie.root, [], 0)]
        
        while queue:
            node, path, depth = queue.pop(0)
            current_path_str = self.make_path_string(path, is_infer=True) if path else "ROOT"
            logger.debug(f"Processing node: {node.token if node.token else 'ROOT'}, current path: {current_path_str}")
            
            # Calculate current threshold based on depth
            current_threshold = threshold * (1 + depth * 0.1)
            logger.debug(f"Current depth: {depth}, threshold: {current_threshold}")
            
            if node.is_end_of_relation():
                relation = self.make_path_string(path, is_infer=True)
                results.append(relation[:-1].replace("·", "."))
                logger.info(f"Found complete relation: {relation[:-1].replace('·', '.')}")
            
            # If current node has no children, continue to next node in queue
            if not node.children:
                logger.debug(f"Node has no children, skipping: {node.token if node.token else 'ROOT'}")
                continue
            
            # Get logits for next token prediction based on current path
            logits = self._get_next_token_logits(question, path,is_mask)
            logger.debug(f"Got logits for next token prediction, shape: {logits.shape}")
            
            # Mask logits based on valid children in the trie
            masked_logits = self._mask_logits_by_trie(logits, node)
            
            # Get probabilities from logits
            probs = torch.softmax(masked_logits, dim=0)
            
            # Find tokens with probability above threshold
            #valid_indices = torch.where(probs > current_threshold)[0]
            valid_indices = torch.where(probs > threshold)[0]
            logger.debug(f"Found {len(valid_indices)} tokens above threshold {current_threshold}")
            
            # If no valid tokens above threshold, don't continue this path
            if len(valid_indices) == 0:
                logger.debug(f"No valid tokens above threshold for path: {current_path_str}, skipping")
                continue
            # Get top-k token indices and their probabilities
            topk_values, topk_indices = torch.topk(
                probs[valid_indices], 
                min(beam_width, len(valid_indices))
            )
            topk_token_ids = valid_indices[topk_indices]
            logger.debug(f"Selected top-{len(topk_token_ids)} tokens with probabilities: {topk_values.tolist()}")
            
            # Convert token IDs to tokens and add to queue
            for i, token_id in enumerate(topk_token_ids):
                token = self.tokenizer.decode(token_id)
                # Remove last character for deduplication
                token_key = token
                logger.debug(f"Considering token: '{token}' with probability: {topk_values[i].item():.4f}")
                
                # Find the corresponding child node
                matches_found = 0
                for child_token, child_node in node.children.items():
                    if child_token.replace("^","").replace("*","")==token_key:
                        # Increment depth when adding to queue
                        queue.append((child_node, path + [child_node], depth + 1))
                        matches_found += 1
                        logger.debug(f"Added child node '{child_token}' to queue")
                
                if matches_found == 0:
                    logger.debug(f"No matching child found for token: '{token}'")
        
        logger.info(f"Route_question completed, found {len(results)} relations")
        return results

    
    def extract_relations_from_question(self, question: str, beam_width: int = 3, 
                                        similarity_threshold: float = 0.4,is_mask:bool=False) -> List[str]:
        """
        High-level function to extract relations from a question.
        
        Args:
            question: The input question
            beam_width: Number of paths to explore at each step
            similarity_threshold: Minimum similarity score to consider a token
            
        Returns:
            List of relations relevant to the question
        """
        logger.info(f"Extracting relations from question: '{question}'")
        try:
            results = self.route_question(question, beam_width, similarity_threshold,is_mask)
            logger.info(f"Extracted {len(results)} relations")
            return results
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
            return []

def extract_relations(question: str, relations: List[str], model, 
                     beam_width: int = 3, similarity_threshold: float = 0.4, device=None) -> List[str]:
    """
    Create a trie from relations and extract relevant relations from the question.
    
    Args:
        question: The input question
        relations: List of relation strings
        model_path_or_name: Path or name of the model
        beam_width: Number of paths to explore at each step
        similarity_threshold: Minimum similarity score to consider a token
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        List of relations relevant to the question
    """
    logger.info(f"Starting extraction with {len(relations)} possible relations")
    
    # Create the trie
    logger.info("Creating token trie...")
    trie = create_token_trie(relations, model.tokenizer)
    logger.info("Token trie created successfully")
    
    # Create the router
    logger.info("Initializing router...")
    router = TrieRouter(trie, model, device=device)
    
    # Extract relations
    logger.info("Extracting relations...")
    result = router.extract_relations_from_question(question, beam_width, similarity_threshold)
    logger.info(f"Extraction complete. Found {len(result)} relations")
    return result


def read_predicates_from_bin(file_path):
    logger.info(f"Reading predicates from binary file: {file_path}")
    docs = []
    
    with open(file_path, "rb") as f:
        # First read the number of entries (int - 4 bytes)
        num_entries_bytes = f.read(4)
        if not num_entries_bytes:
            logger.warning("Empty or invalid binary file")
            return docs
            
        num_entries = int.from_bytes(num_entries_bytes, byteorder='little')
        logger.info(f"Found {num_entries} entries in binary file")
        
        # Read each entry based on the count
        for i in range(num_entries):
            try:
                # Read predicate length
                predicate_length_bytes = f.read(4)
                if not predicate_length_bytes:
                    logger.warning(f"Unexpected end of file after reading {i} entries")
                    break
                    
                predicate_length = int.from_bytes(predicate_length_bytes, byteorder='little')
                
                # Read predicate string
                predicate_bytes = f.read(predicate_length)
                predicate = predicate_bytes.decode()
                
                # Read id (skip it as we don't need it)
                f.read(4)
                
                # Add to our document list
                docs.append(predicate)
                
                if i % 10000 == 0 and i > 0:
                    logger.info(f"Read {i}/{num_entries} predicates...")
                    
            except Exception as e:
                logger.error(f"Error reading entry {i}: {e}")
                break
    
    logger.info(f"Successfully read {len(docs)} predicates")           
    return docs



if __name__ == "__main__":
    # Add command line argument parser
    parser = argparse.ArgumentParser(description="Relation extraction evaluation")
    # Modify model_path argument to point to the adapter directory and make it required
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the fine-tuned adapter directory")
    parser.add_argument('--test_data', type=str, default="cwq_100.json",
                        help="Path to test data JSON file")
    parser.add_argument('--bin_file', type=str, default="edge_label_to_id.bin",
                        help="Path to binary file containing predicates")
    parser.add_argument('--save_file', type=str, default="cwq_test_predictions",
                        help="Path to save file")
    parser.add_argument('--num_samples', type=int, default=-1,
                        help="Number of samples to process")
    parser.add_argument('--threshold', type=float, default=0.1,
                        help="Threshold for relation extraction")
    args = parser.parse_args()

    is_mask_ = "mask" in args.model_path

    output_directory = "../../results/predicted_relation_res/"

    adapter_path = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. read adapter config file
    adapter_config_path = f"{adapter_path}/adapter_config.json"
    logger.info(f"Reading adapter config from: {adapter_config_path}")
    try:
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config["base_model_name_or_path"]
        logger.info(f"Base model path from adapter config: {base_model_path}")
    except FileNotFoundError:
        logger.error(f"Adapter config file not found at {adapter_config_path}")
        exit(1)
    except KeyError:
        logger.error(f"Could not find 'base_model_name_or_path' in {adapter_config_path}")
        exit(1)


    # 2. load tokenizer
    logger.info(f"Loading tokenizer from adapter path: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True) # Added trust_remote_code=True just in case
    logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # 3. load base model from adapter config file
    logger.info(f"Loading base model from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16, # Match fine-tuning settings
        device_map=device,
        trust_remote_code=True # If base model requires it
    )
    logger.info("Base model loaded.")
    logger.info(f"Initial model embedding size: {model.get_input_embeddings().weight.shape}")

    
    logger.info(f"Resizing model token embeddings to: {len(tokenizer)}")
    if(len(tokenizer) != model.get_input_embeddings().weight.shape[0]):
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embedding size: {model.get_input_embeddings().weight.shape}")
    else:
        logger.info(f"Model embedding size is already the same as tokenizer size: {model.get_input_embeddings().weight.shape}")

    # 5. load PEFT adapter
    logger.info(f"Loading adapter weights from: {adapter_path}")
    # Note: The first argument is the already loaded and resized model object
    model = PeftModel.from_pretrained(model, adapter_path)
    logger.info("Adapter loaded successfully.")
    
    logger.info("Starting main evaluation process")
    queries = []
    docs = []
    gt_edges = []
    
    
    with open(args.test_data, "r") as f:
        data = json.load(f)
        if('cwq' in args.test_data.lower() or 'complex' in args.test_data.lower()):
            num_samples = args.num_samples if args.num_samples > 0 else len(data)
            for item in data[:num_samples]:
                if(item.get("sparql") is not None):
                    # start processing
                    sparql_query = item["sparql"]
                    query = item["question"]
                    pattern = r"ns:(\S+)"
                    matches = re.findall(pattern, sparql_query)
                    matched_entity = [item_ for item_ in matches if item_[1]=='.' and item_.count('.')==1 ]
                    matched_relation = [item_ for item_ in matches if (item_ not in matched_entity) and (item_.startswith("common.")==False) ]
                    queries.append(query)
                    gt_edges.append(matched_relation)
        elif('webqsp' in args.test_data.lower()):
            num_samples = args.num_samples if args.num_samples > 0 else len(data["Questions"])
            process_data = data["Questions"][0:num_samples] if num_samples != -1 else data["Questions"]
            for item in process_data:
                queries.append(item["RawQuestion"])
                if item.get("Parses") is not None:
                    parses = item["Parses"]
                    current_gt = []  # This will be a list of lists for multiple ground truths
                    for parse in parses:
                        if parse.get("Sparql") is not None:
                            sparql_query = parse["Sparql"]
                            pattern = r"ns:(\S+)"
                            matches = re.findall(pattern, sparql_query)
                            matched_entity = [item_ for item_ in matches if item_[1]=='.' and item_.count('.')==1 ]
                            matched_relation = [item_ for item_ in matches if (item_ not in matched_entity) and (item_.startswith("common.")==False) ]
                            if matched_relation:  # Only add non-empty relation lists
                                current_gt.append(matched_relation)
                    # If no valid parses found, add empty list
                    if not current_gt:
                        current_gt = [[]]
                    gt_edges.append(current_gt)
        elif('grailqa' in args.test_data.lower()):
            num_samples = args.num_samples if args.num_samples > 0 else len(data)
            for item in data[:num_samples]:
                queries.append(item["question"])
                sparql_query = item["sparql_query"]
                pattern = r":([\w\.]+)\s"
                matches = re.findall(pattern, sparql_query)
                matched_entity = [item_ for item_ in matches if item_.count('.')<=1] # :m.*****, skip
                matched_relation = [item_ for item_ in matches if (item_ not in matched_entity) and (item_.startswith("common.")==False) and (item_.startswith("type.")==False)] # evict : and the last space
                gt_edges.append(matched_relation)
        else:
            raise ValueError(f"Unsupported dataset: {args.test_data}")

    logger.info(f"Loaded {len(queries)} test questions")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info("Reading predicates from binary file")
    relations = read_predicates_from_bin(args.bin_file)
    logger.info(f"Creating trie with {len(relations)} relations")
    trie = create_token_trie(relations, tokenizer)
    logger.info("Initializing router")
    router = TrieRouter(trie, model, device=device) 
    logger.info("Beginning evaluation on test queries")
    
    # Track timing information
    query_times = []
    start_time_total = time.time()
    
    # Create a copy of the data to add predictions
    #output_data = data[:100]
    if('webqsp' in args.test_data.lower() or 'freebaseqa' in args.test_data.lower()):
        output_data = data["Questions"][0:num_samples] if num_samples != -1 else data["Questions"]
    else:
        output_data = data[:num_samples]
    
    for i, query in enumerate(queries):
        logger.info(f"\n===== Query {i+1}/{len(queries)} =====")
        logger.info(f"Query: {query}")
        
        try:
            # Start timing this query
            start_time_query = time.time()
            
            results = router.extract_relations_from_question(query, beam_width=5, similarity_threshold=args.threshold,is_mask=is_mask_)
            
            # Save predicted relations to the output data
            output_data[i]["predicted_relation"] = results
            
            # End timing and record
            query_time = time.time() - start_time_query
            query_times.append((i+1, query, query_time))
            
            logger.info(f"Query processing time: {query_time:.4f} seconds")
            logger.info(f"Ground truth relations: {gt_edges[i]}")
            logger.info(f"Predicted relations: {results}")

            # Get predicted relations from top results
            pred_relations = set(results)
        except Exception as e:
            logger.error(f"Error processing query {i+1}: {e}")
            logger.info("Skipping this query and continuing with the next one")
            # Add empty prediction for failed queries
            output_data[i]["predicted_relation"] = []
    
    # Save the output data with predictions to a new JSON file
    
    output_file = output_directory + args.save_file + ".json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved predictions to {output_file}")
    # Calculate total elapsed time
    total_time = time.time() - start_time_total
    query_times.sort(key=lambda x: x[2], reverse=True)