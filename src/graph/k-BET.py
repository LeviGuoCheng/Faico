from KnowledgeBase import KnowledgeBase
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
import json
import os
from util import process_dataset_for_kBET
from tqdm import tqdm
from typing import List, Dict, Set, Tuple
# from ListTrie import ListPrefixChecker
from config import FREEBASE_DIR


depth = None 
MAX_TRIPLETS = 1e2

# There are cases (not very common) where kBET is still too large
# we calculate a minimal connected subgraph
def find_minimal_connected_subgraph(G, T, M):
    if not all(u in G for u in T):
        T_ = []
        for t in T:
            if t in G:
                T_.append(t)
        T = T_
    edge_labels = {data['relation'] for _, _, data in G.edges(data=True) if 'relation' in data}
    # relations = nx.get_edge_attributes(G, 'relation')
    # edges_labels = set(relations.values())
    missing_labels = set(M) - edge_labels
    if missing_labels:
        return G
    
    if len(T) == 1 and not M:
        subgraph = nx.Graph()
        subgraph.add_node(list(T)[0])
        return subgraph
    elif len(T)==1 and M:
        subgraph = nx.Graph()
        subgraph.add_node(list(T)[0])
    else:
        # len(T)>1
        terminal_nodes = list(T)
        frozen_subgraph = steiner_tree(G, terminal_nodes)
        subgraph = nx.Graph(frozen_subgraph)


    contained_labels = {data['relation'] for _, _, data in subgraph.edges(data=True) if 'relation' in data}
    # subgraph_relations = nx.get_edge_attributes(subgraph, 'relation')
    # contained_labels = set(subgraph_relations.values())
    missing_labels = set(M) - contained_labels
    
    while missing_labels:
        label = missing_labels.pop()
        
        edges_with_label = [(u, v, data) for u, v, data in G.edges(data=True) 
                           if 'relation' in data and data['relation'] == label]
        
        if not edges_with_label:
            raise ValueError(f"No edge with label {label} exists in the graph")
        
        best_edge = None
        min_cost = float('inf')
        
        for u, v, data in edges_with_label:
            if u in subgraph and v in subgraph:
                best_edge = (u, v, data)
                min_cost = 0
                break
            elif u in subgraph or v in subgraph:
                if min_cost > 1:
                    min_cost = 1
                    best_edge = (u, v, data)
            else:
                cost_u = float('inf')
                cost_v = float('inf')
                
                for sg_node in subgraph:
                    try:
                        dist_u = nx.shortest_path_length(G, u, sg_node)
                        cost_u = min(cost_u, dist_u)
                        
                        dist_v = nx.shortest_path_length(G, v, sg_node)
                        cost_v = min(cost_v, dist_v)
                    except nx.NetworkXNoPath:
                        continue
                total_cost = min(cost_u, cost_v) + 1
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_edge = (u, v, data)
        
        if best_edge:
            u, v, data = best_edge
            for node in [u, v]:
                if node not in subgraph:
                    min_path_length = float('inf')
                    best_path = None
                    
                    for sg_node in subgraph:
                        try:
                            path = nx.shortest_path(G, node, sg_node)
                            if len(path) < min_path_length:
                                min_path_length = len(path)
                                best_path = path
                        except nx.NetworkXNoPath:
                            continue
                    
                    if best_path:
                        for i in range(len(best_path)-1):
                            n1, n2 = best_path[i], best_path[i+1]
                            edge_data = G.get_edge_data(n1, n2)
                            subgraph.add_node(n1)
                            subgraph.add_node(n2)
                            subgraph.add_edge(n1, n2, **edge_data)
            
            subgraph.add_edge(u, v, **data)
            
            contained_labels = {data['relation'] for _, _, data in subgraph.edges(data=True) if 'relation' in data}
            missing_labels = set(M) - contained_labels
    return subgraph



def is_dominated(budget_a: dict, budget_b: dict) -> bool:
    """
    Determine whether budget_a is dominated by budget_b (budget_a ⪯ budget_b).
    
    According to Definition 3:
    If for all keys, we have budget_a[key] <= budget_b[key],
    return True; otherwise return False.
    
    Parameters:
    budget_a -- The current path's budget (dict: {str: int})
    budget_b -- The comparison target (usually the historically optimal budget 
                of a previously visited node) (dict: {str: int})
    """
    keys = budget_a.keys() 
    
    for k in keys:
        val_a = budget_a.get(k, 0)
        val_b = budget_b.get(k, 0)
        
        # If any dimension of a is greater than b, then a cannot be fully 
        # dominated by b
        if val_a > val_b:
            return False
            
    # If all dimensions satisfy a <= b, then a is dominated by b
    return True

# we use a stack-based data structure to traverse and store the k-BET
class kBETGraph:
    def __init__(self, allowed_items, k=1):
        """
        Initialize the graph.

        Parameters:
        allowed_items -- the set of elements allowed to be pushed (list, set, or any iterable)
        k -- int, the maximum allowed occurrences for each element (i.e., k), default = 1
        """
        self.stack = [] #current k-BET path
        self.graph = nx.Graph()
        self.budget_dict:Dict[str, List[Dict[str, int]]]={}
        
        # Convert the allowed item set into a dictionary where the key is the element
        # and the value is the remaining allowed occurrences.
        # Example: {'relation_A': k, 'relation_B': k}
        self.budget = {item: k for item in allowed_items}
        
        # Initial history record; note that a copy must be stored to avoid later modifications
        # affecting the initial state.
        self.budget_history = [self.budget.copy()]
        
        # Save the initial configuration for reset
        self.initial_budget = self.budget.copy()


    def push(self, item):
        """
        Push an element (triplet) onto the graph if budget allows
        and the current budget is not dominated by previously seen budgets for obj_vtx.
        """

        if not isinstance(item, tuple) or len(item) != 3:
            return False

        sub_vtx = item[0]
        label = item[1]
        obj_vtx = item[2]

        # Check budget availability
        current_count = self.budget.get(label, 0)
        if current_count <= 0:
            return False

        # Case 1: obj_vtx first time visited, and we have budget, add
        if obj_vtx not in self.budget_dict:
            self.budget_dict[obj_vtx] = []

            # update stack / budget / history
            self.stack.append(label)
            self.budget[label] = current_count - 1

            # add edge
            self.graph.add_edge(sub_vtx, obj_vtx, relation=label)

            # save history
            new_budget = self.budget.copy()
            self.budget_history.append(new_budget)

            # save to budget_dict
            self.budget_dict[obj_vtx].append(new_budget)

            return True

        # Case 2: obj_vtx has been visited: check dominance

        # if we add current triple, self.budget -> cur_budget
        cur_budget = self.budget.copy()
        cur_budget[label] = current_count -1

        # check if current budget is dominated by old budget
        for old_budget in self.budget_dict[obj_vtx]:
            if is_dominated(cur_budget, old_budget):
                # current budget is worse or equal → prune
                return False
            
        # Otherwise, current budget is better on some dimension → keep exploring
        self.stack.append(label)
        self.budget[label] = current_count - 1

        # add edge
        self.graph.add_edge(sub_vtx, obj_vtx, relation=label)

        # save new budget
        new_budget = self.budget.copy()  
        self.budget_history.append(new_budget) 

        # IMPORTANT: append/update the *current budget copy* to budget_dict
        
        # update
        self.budget_dict[obj_vtx] = [
            old_budget
            for old_budget in self.budget_dict[obj_vtx]
            if not is_dominated(old_budget, new_budget)
        ]
        # append
        self.budget_dict[obj_vtx].append(new_budget)

        return True
    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.stack) == 0
        
    def pop(self):
        """
        Pop the top element from the stack and restore its allowed occurrence count.

        Returns:
        The popped element; raises IndexError if the stack is empty.
        """
        if self.is_empty():
            raise IndexError("Pop from an empty stack")
        
        item = self.stack.pop()
        
        # State rollback logic
        self.budget_history.pop()  # Remove the current state (state before pop)
        
        # Core change: restore the dictionary to the previous state copy
        # Must use copy(), otherwise future pushes would modify historical states.
        self.budget = self.budget_history[-1].copy()
        
        return item

    def get_cur_feasible(self) -> set[str]:
        return {label for label, remain in self.budget.items() if remain > 0}


"""
@para t_nodes: the topic entites for the query to begin
@para gt_edge: groundtruth edges that only used in
@para KB: the knowledge base used for traversing

"""			
def explore(t_node: str, KB: KnowledgeBase, gs: kBETGraph):
    """Depth-first search traversal of knowledge graph"""
 
    # violate the kBEt path constraint, return
    if len(gs.stack)>=depth:
            return
    # print(f"\nProcessing entity: {t_node}")
    # print(f"Current stack size: {len(gs.stack)}", gs.stack)
    # print(f"Number of visited triplets: {len(visited)}")
    
    # print(f"Retrieving relations for entity {t_node}...")
    edges_dict = KB.GetEdgeLabel(t_node)
    edges = {edge["relation"] for edge in edges_dict}

    # print(f"Found {len(edges)} relations")
    
    # print(f"Relations to be explored: {len([edge for edge in edges if edge in gs.allowed_items])}")
    
    # print(edges)
    for edge in edges:
        if edge in gs.get_cur_feasible() :

            triplets = KB.GetVertex(t_node, edge)

            # When too many nodes are found, try using two-hop search
            if len(triplets) > 10*MAX_TRIPLETS:
                continue
            if len(triplets) > MAX_TRIPLETS and len(gs.stack)<=(depth-2):#!TODO Check if the size needs to be adjusted
                allowed_items = gs.get_cur_feasible()
                feasible_next_edges = {next_edge for next_edge in allowed_items if next_edge != edge}
                in_triplets_ = KB.GetOutVertex(t_node, edge)
                for t1 in in_triplets_:
                    for next_edge in feasible_next_edges:
                        triplets_ = KB.GetVertex(t1[2], next_edge)
                        for t2 in triplets_:
                            if gs.push(t1):
                                if gs.push(t2):
                                    explore(t2[2], KB, gs)
                                    gs.pop()  # Pop the second hop
                                else:
                                    gs.pop()  # Pop the first hop
            else:
                for triplet in triplets:
                    if gs.push(triplet):
                        explore(triplet[2], KB, gs)
                        gs.pop()
    return


"""
@para t_nodes: the topic entites for the query to begin
@para gt_edge: groundtruth edges that only used in
@para KB: the knowledge base used for traversing
@return gt_stack.graph: groundtruth graph that used in reasoning
"""
def kBETRetrieval(t_nodes:list[str], pd_edges:list[str],KB:KnowledgeBase,k=1):
    gs = kBETGraph(pd_edges,k)
    remaining_nodes = set(t_nodes)
    for node in remaining_nodes:
        explore(node,KB,gs)
    return gs.graph



def process_one_question(data_,value,id=None,k=1):
    """
    @param data_: input file for predicted relations
    @param value: data entry
    @param id: data ID
    @return G: knowledge graph
    @return value: data entry
    """
    pd_edges = None
    for item in data_:
        if item["ID"] == id:
            pd_edges = item["predicted_relation"]
            break
    if pd_edges is None:
        return None,value
    t_nodes = list(value["topic_entity"].keys())    
    pd_edges = [i for i in pd_edges if i != "common.topic.notable_types"] # Filter out unnecessary relations

    print("Predicted edges:" ,pd_edges)

    import time
    G = kBETRetrieval(t_nodes, pd_edges, KB, k)
    triplets = [(u, d['relation'], v) for u, v, d in G.edges(data=True)]

    used_relation = {_ for u, _, v in triplets}
    value["pre_triplets"] = triplets
    # print(f"Number of extracted triplets: {len(triplets)}")
    
    print(value["question"])

    if(len(triplets)>20):
        G = find_minimal_connected_subgraph(G,set(t_nodes),list(used_relation))
        
    print("Question:",value["question"])
    triplets = [(u, d['relation'], v) for u, v, d in G.edges(data=True)]
    # Print the number of triplets
    # print(f"Number of extracted triplets: {len(triplets)}")
    # print(f"Triplet size: {len(triplets)}") 
    value["triplets"] = triplets

    return G,value




def generate_experiment_dir(base_dir="../../results/k-BET_res/"):
    from datetime import datetime
    import os
    
    today = datetime.now().strftime("%m_%d")
    base_dir = os.path.join(current_dir, base_dir)
    
    existing_exps = []
    for dir_name in os.listdir(base_dir):
        if dir_name.startswith(today + "_exp"):
            existing_exps.append(dir_name)
    
    
    exp_num = len(existing_exps) + 1
    
    
    new_dir_name = f"{today}_exp{exp_num}"
    new_dir_path = os.path.join(base_dir, new_dir_name)
    
    
    os.makedirs(new_dir_path, exist_ok=True)
    
    return new_dir_path





# ... existing code ...
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)#relative path
KB = KnowledgeBase(FREEBASE_DIR)


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True ,help="the input file path with the predicted relations")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save results, default will be save in results/k-BET_res/")
    parser.add_argument("--description", type=str, default="a test",help="a text description for the experiment")
    parser.add_argument("--dataset", type=str, required=True ,default="cwq")
    parser.add_argument("--depth", type=int, required=True ,help="the max depth for exploring, 4 for grailqa and cwq, 2 for webqsp")
    parser.add_argument("--k", type=int, default=1, help="the max number of edge label")
    args = parser.parse_args()
    input_file = args.input
    k = args.k
    depth = args.depth
    if args.dataset == "cwq":
        cwq_dataset = os.path.join(current_dir, "../../data/dataset/cwq.json")
        with open(cwq_dataset, "r") as ff:
            data = json.load(ff)
    elif args.dataset == "webqsp":
        webqsp_dataset = os.path.join(current_dir, "../../data/dataset/WebQSP.json")
        with open(webqsp_dataset, "r") as ff:
            data = json.load(ff)
    elif args.dataset == "grailqa":
        grailqa_dataset = os.path.join(current_dir, "../../data/dataset/grailqa.json")
        with open(grailqa_dataset, "r") as ff:
            data = json.load(ff)

    data = process_dataset_for_kBET(data,args.dataset)


    with open(input_file, "r") as fr:
        input_data = json.load(fr)

    input_data = process_dataset_for_kBET(input_data,args.dataset)#!TODO

    data_to_write = {}
    if args.save_dir == "":
        save_dir = generate_experiment_dir()  
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    description_file_name = "description.txt"
    description_path = os.path.join(save_dir, description_file_name)
    with open(description_path, "w") as f:
        f.write(args.description)



    import time
    time_start = time.time()
    for dit in tqdm(data):
        value = dit
        key = value["ID"]
        start_time_ = time.time()
        

        G,value = process_one_question(input_data,value,key,k)
        if G is None:
            continue
        value["time"] = time.time() - start_time_
        data_to_write[key] = value
        
        #Get the absolute path of the current code
        
    time_end = time.time()
    
    with open(description_path, "a") as f:
        f.write(f"\nk-BET time consumption: {time_end - time_start}s")
    
    save_file_name = "k-BET_result.json"
    save_path = os.path.join(save_dir, save_file_name)
    
    with open(save_path,"w") as fff:
        json.dump(data_to_write,fff,indent=4)
    
    
    
