from transformers import AutoTokenizer
from typing import List, Dict, Optional
import json
import torch
import argparse

class TrieNode:
    def __init__(self, token: str = None):
        # special token with 
        # "^" suffic to indicate split token
        # "*" suffix to indicate non-split token
        # "*" prefix to indicate the end of a token
        # "^" prefix to indicate end of relation
        self.token = token
        self.children: Dict[str, TrieNode] = {}
        #self.is_end_of_relation = False
        #self.is_end_of_token = False
        #self.is_tokenizer_split = None  # New attribute to track tokenizer splits
        #self.is_non_split = None # This is used to handle the special case where token is structural ambigious, e.g. "people.birth" and "people.birth.xxx"
    def is_end_of_relation(self) -> bool:
        return False if (self.token is None) else (self.token.startswith("^"))
    def is_end_of_token(self) -> bool:
        return self.token.startswith("*") or self.token.startswith("^")
    def is_split_token(self) -> bool:
        return self.token.endswith("^")
    def is_non_split_token(self) -> bool:
        return self.token.endswith("*")

class TokenTrie:
    def __init__(self, tokenizer):
        self.root = TrieNode()  # Root node is NULL
        self.tokenizer = tokenizer
        self.tokenize_cache = {}  # Cache for tokenization results
    
    def _should_split_token(self, token: str) -> bool:
        """Check if a token should be split based on tokenizer output."""
        if token in self.tokenize_cache:
            tokens = self.tokenize_cache[token]
        else:
            tokens = self.tokenizer.tokenize(token)
            self.tokenize_cache[token] = tokens
        return len(tokens) > 1
    
    def _split_by_tokenizer(self, token: str) -> List[str]:
        """Split a token into subtokens using the tokenizer."""
        if token in self.tokenize_cache:
            return self.tokenize_cache[token]
        tokens = self.tokenizer.tokenize(token)
        self.tokenize_cache[token] = tokens
        return tokens
    
    def insert(self, relation: str) -> None:
        """Insert a relation into the trie."""
        # Split the relation by dots
        parts = relation.split('.')
        current = self.root

        #To avoid structural ambigious, we add "^" to the subtoken and "*" to distinguish the parts
        # this is mainly used for string reconstruction given the ancestor path
        for part in parts:
            # Check if the token contains underscores and should be split
            if self._should_split_token(part):
                # Process each subtoken
                subtokens = self._split_by_tokenizer(part)
                len_subtokens = len(subtokens)
                for i, subtoken in enumerate(subtokens):
                    subtoken = subtoken+"^"
                    if i==len_subtokens-1:
                        subtoken = "*"+subtoken
                    if subtoken not in current.children:
                        current.children[subtoken] = TrieNode(subtoken)
                            #current.children[subtoken] = TrieNode(subtoken)    
                            #current.children[subtoken].is_end_of_token = True
                            #continue 
                    #     current.children[subtoken].is_tokenizer_split = True
                    # elif (current.children[subtoken].is_non_split) and (current.children[subtoken].is_tokenizer_split is None):
                    #     current.children[subtoken].is_tokenizer_split = True
                    current = current.children[subtoken]
            else:
                # Keep the token as is
                part = part+"*"
                if part not in current.children:
                    current.children[part] = TrieNode(part)
                    #current.children[part].is_end_of_token = True
                #     current.children[part].is_non_split = True
                # elif (current.children[part].is_tokenizer_split) and (current.children[part].is_non_split is None):
                #     current.children[part].is_non_split = True
                current = current.children[part]
        
        # Mark the end of a relation
        # this aims to distinguish "_place" between people.birth_place and people.birth_place.location
        if current.token.startswith("*"):
            current.token = "^" + current.token[1:]
        else:
            current.token = "^" + current.token
    
    def build_from_relations(self, relations: List[str]) -> None:
        """Build trie from a list of relations."""
        for relation in relations:
            self.insert(relation)
    
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

    def _print_all_relations(self, node: Optional[TrieNode] = None, current_path:List[TrieNode] = None) -> None:
        """Helper method to print all complete relations in the trie."""
        if node is None:
            node = self.root
        if current_path is None:
            current_path = []
        
        # Skip if no children and not end of relation
        if not node.children and not node.is_end_of_relation():
            return
        #new_path = current_path.copy()
        # is not none is for the case where the token is the root node
        if node.token is not None:
            if node.is_end_of_relation():
                # Construct the path string
                #path_string = current_path+node.token[:-1]
                print(self.make_path_string(current_path)) 
        
        # Add children to the queue
        for _, child_node in node.children.items():
            self._print_all_relations(child_node, current_path+[child_node])
    
    def to_dict(self) -> Dict:
        """Convert the trie to a dictionary representation."""
        def node_to_dict(node):
            result = {
                "token": node.token,
                "is_end": node.is_end_of_relation(),
                "children": {}
            }
            for token, child in node.children.items():
                result["children"][token] = node_to_dict(child)
            return result
        
        return node_to_dict(self.root)
    
    def get_train_data(self, query: str,is_mask:bool=False) -> List[dict]:
        """
        Get training data from the trie for fine-tuning LLM.
        
        Generates training data by traversing the trie in BFS order, creating examples
        where each node's children are potential completions with equal probability.
        
        Args:
            query: The question to use in the training examples
            
        Returns:
            List of dictionaries containing training examples in the format:
            {
                "text": "Generate relation for question → {query}\nAnswer: {ancestors}",
                "target_probs": {child_token1: prob1, child_token2: prob2, ...}
            }
        """
        training_data = []
        
        # BFS traversal
        queue = [(self.root, [])]  # (node, path_string, ancestors)
        
        while queue:
            current_node, path = queue.pop(0)
            
            # Skip if no children
            if not current_node.children:
                continue
            current_tokens = []
            for token, _ in current_node.children.items():
                current_tokens.append(token.replace("^","").replace("*",""))
            current_tokens = list(set(current_tokens))
            # Calculate equal probability for each child
            child_count = len(current_tokens)
            # TODO: alter to semantic probability (using embedding similarity?) instead of equal probability
            child_prob = 1.0 / child_count
            
            # Create target probabilities dictionary
            target_probs = {}
            for token in current_tokens:
                target_probs[token] = child_prob
                #target_probs[token] = 1.0
            
            # Construct the ancestor path string
            path_string = self.make_path_string(path,is_infer=True)
            if is_mask:
                path_string = path_string + "<mask>"
            #people->birth_place  -|
            #                      |--- special case where the parent token is structural ambigious
            #people->birth        -|    the first birth is "End" , the second birth is "splitted"
            #people->name
            
            # Create the training example
            # example = {
            #     "text": f"Generate relation for question -> {query}\nAnswer: {path_string}<mask>",
            #     "target_probs": target_probs
            # }
            example = {
                "text": f"Generate relation for question -> {query}\nAnswer: {path_string}",
                "target_probs": target_probs
            }
            
            training_data.append(example)
            
            # Add children to the queue
            for token, child_node in current_node.children.items():
                #new_path_string = path_string + token
                new_path= path + [child_node]
                queue.append((child_node, new_path))
        
        return training_data
        

def create_token_trie(relations: List[str], tokenizer) -> TokenTrie:
    """
    Create a token trie from a list of relations.
    
    Args:
        relations: List of relation strings (e.g., ["people.birth.birth_place"])
        model_path_or_name: Path or name of the tokenizer model
    
    Returns:
        A TokenTrie instance with the relations inserted
    """
    trie = TokenTrie(tokenizer)
    trie.build_from_relations(relations)
    return trie

# Example
if __name__ == "__main__":
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="TokenTrie example")
    parser.add_argument('--tokenizer_path', type=str, 
                        default="/data/chengguo/git_repo/Meta-Llama-3.1-8B-Instruct",
                        help="Path to the tokenizer model")
    args = parser.parse_args()
    
    # Example relations
    example_relations = [
        "people.birth_place.birth_place",
        "people.birth.xxx",
        "people.birth_place",
        "people.birth.place",
        "people.profession",
        "organization.founder",
        "exhibitions.location",
        "location.country.capital_city"
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # Create a trie
    trie = create_token_trie(example_relations, tokenizer)
    
    # Print the trie structure
    print("Trie Structure:")
    trie._print_all_relations()
    
    # Export as JSON for visualization or further use
    # trie_dict = trie.to_dict()
    # print("\nTrie as JSON:")
    # print(json.dumps(trie_dict, indent=2))
