import json
import argparse


def jsonl_to_json(filename):
    """
    Convert jsonl file to a list in json format
    
    Parameters:
        filename (str): Path to the jsonl file
        
    Returns:
        list: List containing all json objects
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


def align_for_baseline(filename):
    if "jsonl" in filename:
        data = jsonl_to_json(filename)
    else:
        with open(filename, "r") as f:
            data = json.load(f)    
    return data



def eval_answer_f1_cwq(pred_data,dataset_dict):

    p_list = []
    r_list = []
    f_list = []
    hit_list = []
    p_dict = {}
    r_dict = {}
    f_dict = {}
    hit_dict = {}
    acc_num = 0

    pred_dict = {}
    acc_qid_list = [] # Pred Answer ACC
    for key, pred in pred_data.items():
        qid = key
        pred_answer = set(pred["predict_answers"])
        pred_dict[qid]=pred_answer
    
        example = dataset_dict[qid]

        gt_answer = set(example['answer'])
        

        pred_answer = set(pred_dict.get(qid,{}))

        if pred_answer == gt_answer:
            acc_num+=1
            acc_qid_list.append(qid)

        if len(pred_answer)== 0:
            if len(gt_answer)==0:
                p=1
                r=1
                f=1
                hit=1
            else:
                p=0
                r=0
                f=0
                hit=0
        elif len(gt_answer)==0:
            p=0
            r=0
            f=0
            hit=0
        else:
            p = len(pred_answer & gt_answer)/ len(pred_answer)
            r = len(pred_answer & gt_answer)/ len(gt_answer)
            f = 2*(p*r)/(p+r) if p+r>0 else 0
            hit = 1 if len(pred_answer & gt_answer)>0 else 0
        

        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
        hit_list.append(hit)
        p_dict[qid] = p
        r_dict[qid] = r
        f_dict[qid] = f
        hit_dict[qid] = hit
    
    p_average = sum(p_list)/len(p_list)
    r_average = sum(r_list)/len(r_list)
    f_average = sum(f_list)/len(f_list)
    hits1 = sum(hit_list)/len(hit_list)

    res = f'Total: {len(p_list)}, Hit@1: {hits1:.3f}, PRE: {p_average:.3f}, REC: {r_average:.3f}, F1: {f_average:.3f}'
    print(res)



def eval_answer_f1_grailqa(pred_data,dataset_dict):
    p_list = []
    r_list = []
    f_list = []
    hit_list = []
    p_dict = {}
    r_dict = {}
    f_dict = {}
    hit_dict = {}
    acc_num = 0

    pred_dict = {}
    acc_qid_list = [] # Pred Answer ACC
    for key, pred in pred_data.items():
        if isinstance(key,str) and key.isdigit():
            key =eval(key)
        qid = key
        pred_answer = set(pred["predict_answers"])
        pred_dict[qid]=pred_answer
    
        example = dataset_dict[qid]
        # gt_sparql = example['sparql']

        answer_list = []
        answers = example["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])

        gt_answer = set(answer_list)
        

        pred_answer = set(pred_dict.get(qid,{}))

        if pred_answer == gt_answer:
            acc_num+=1
            acc_qid_list.append(qid)

        if len(pred_answer)== 0:
            if len(gt_answer)==0:
                p=1
                r=1
                f=1
                hit=1
            else:
                p=0
                r=0
                f=0
                hit=0
        elif len(gt_answer)==0:
            p=0
            r=0
            f=0
            hit=0
        else:
            p = len(pred_answer & gt_answer)/ len(pred_answer)
            r = len(pred_answer & gt_answer)/ len(gt_answer)
            f = 2*(p*r)/(p+r) if p+r>0 else 0
            hit = 1 if len(pred_answer & gt_answer)>0 else 0
        


        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
        hit_list.append(hit)
        p_dict[qid] = p
        r_dict[qid] = r
        f_dict[qid] = f
        hit_dict[qid] = hit
    
    p_average = sum(p_list)/len(p_list)
    r_average = sum(r_list)/len(r_list)
    f_average = sum(f_list)/len(f_list)
    hits1 = sum(hit_list)/len(hit_list)

    res = f'Total: {len(p_list)}, Hit@1: {hits1:.3f}, PRE: {p_average:.3f}, REC: {r_average:.3f}, F1: {f_average:.3f}'
    print(res)

def eval_answer_f1_webqsp(predAnswers,goldData):

    def FindInList(entry,elist):
        for item in elist:
            if entry == item:
                return True
        return False
    def CalculatePRF1(goldAnswerList, predAnswerList):
        if len(goldAnswerList) == 0:
            if len(predAnswerList) == 0:
                return [1.0, 1.0, 1.0, 1]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
            else:
                return [0.0, 1.0, 0.0, 1]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
        elif len(predAnswerList)==0:
            return [1.0, 0.0, 0.0, 0]    # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
        else:
            
            glist = [x["AnswerArgument"] if x["AnswerType"] == "Value" else x["EntityName"] for x in goldAnswerList]
            plist =predAnswerList

            tp = 1e-40  # numerical trick
            fp = 0.0
            fn = 0.0

            for gentry in glist:
                if FindInList(gentry,plist):
                    tp += 1
                else:
                    fn += 1
            for pentry in plist:
                if not FindInList(pentry,glist):
                    fp += 1


            precision = tp/(tp + fp)
            recall = tp/(tp + fn)
            
            f1 = (2*precision*recall)/(precision+recall)
            
            if tp > 1e-40:
                hit = 1
            else:
                hit = 0
            return [precision, recall, f1, hit]
    
    PredAnswersById = {}

    for key,item in predAnswers.items():
        PredAnswersById[key] = item["predict_answers"]

    total = 0.0
    f1sum = 0.0
    recSum = 0.0
    precSum = 0.0
    hitSum = 0
    numCorrect = 0
    prediction_res = []
    if "Questions" in goldData:
        goldData = goldData["Questions"]
    for entry in goldData:
        # if entry["QuestionId"] not in PredAnswersById: #!TODO
        #     continue
        skip = True
        for pidx in range(0,len(entry["Parses"])):
            np = entry["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False

        if(len(entry["Parses"])==0 or skip):
            continue

        total += 1
    
        id = entry["QuestionId"]
    
        if id not in PredAnswersById:
            print("The problem " + id + " is not in the prediction set")
            print("Continue to evaluate the other entries")
            continue

        if len(entry["Parses"]) == 0:
            print("Empty parses in the gold set. Breaking!!")
            break

        predAnswers = PredAnswersById[id]

        bestf1 = -9999
        bestf1Rec = -9999
        bestf1Prec = -9999
        besthit = 0

        for pidx in range(0,len(entry["Parses"])):
            pidxAnswers = entry["Parses"][pidx]["Answers"]
            prec,rec,f1,hit = CalculatePRF1(pidxAnswers,predAnswers)
            if f1 > bestf1:
                bestf1 = f1
                bestf1Rec = rec
                bestf1Prec = prec
            if hit > besthit:
                besthit = hit

        f1sum += bestf1
        recSum += bestf1Rec
        precSum += bestf1Prec
        hitSum += besthit

        pred = {}
        pred['qid'] = id
        pred['precision'] = bestf1Prec
        pred['recall'] = bestf1Rec
        pred['f1'] = bestf1
        pred['hit'] = besthit
        prediction_res.append(pred)

        if bestf1 == 1.0:
            numCorrect += 1
        

    print("Number of questions:", int(total))
    print("Hits@1 over questions: %.3f" % (hitSum / total))
    print("Average precision over questions: %.3f" % (precSum / total))
    print("Average recall over questions: %.3f" % (recSum / total))
    print("Average f1 over questions (accuracy): %.3f" % (f1sum / total))


def eval_answer_f1_router(predict_data,dataset):
    if dataset == "webqsp":
        with open("../../data/dataset/WebQSP.json", "r") as f:
            dataset_data = json.load(f)
        eval_answer_f1_webqsp(predict_data,dataset_data)
    elif dataset == "cwq":
        with open("../../data/dataset/cwq.json", "r") as f:
            dataset_data = json.load(f)
        process_dataset = {}
        for item in dataset_data:
            process_dataset[item["ID"]] = item
        eval_answer_f1_cwq(predict_data,process_dataset)
    elif dataset == "grailqa":
        with open("../../data/dataset/grailqa.json","r") as f:
            dataset_data = json.load(f)
        process_dataset = {}
        for item in dataset_data:
            process_dataset[item["qid"]] = item
        eval_answer_f1_grailqa(predict_data,process_dataset)
    
    

def eval_containment_hit(output_datas,dataset):
    def align(dataset_name, ID_string, key, ground_truth_datas):
        answer_list= []
        if isinstance(key,str) and key.isdigit():
            key =eval(key)
        origin_data = [j for j in ground_truth_datas if j[ID_string] == key][0]
        if dataset_name == 'cwq':
            if 'answers' in origin_data:
                answers = origin_data["answers"]
            else:
                answers = origin_data["answer"]
            if type(answers) == list:
                answer_list.extend(answers)
            else:
                answer_list.append(answers)

        elif dataset_name == 'webqsp':
            answers = origin_data["Parses"]
            for answer in answers:
                for name in answer['Answers']:
                    if name['EntityName'] == None:
                        answer_list.append(name['AnswerArgument'])
                    else:
                        answer_list.append(name['EntityName'])
        elif dataset_name == 'grailqa':
            answers = origin_data["answer"]
            for answer in answers:
                if "entity_name" in answer:
                    answer_list.append(answer['entity_name'])
                else:
                    answer_list.append(answer['answer_argument'])
        return list(set(answer_list))
    

    def exact_match(response, answers):
        if response == "":
            return False

        clean_result = response.strip().replace(" ","").lower()
        for answer in answers:
            clean_answer = answer.strip().replace(" ","").lower()
            if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
                return True
        return False


    if dataset == "webqsp":
        with open("../../data/dataset/WebQSP.json", "r") as f:
            ground_truth_datas = json.load(f)
            ID_string = 'QuestionId'
    elif dataset == "cwq":
        with open("../../data/dataset/cwq.json","r") as f:
            ground_truth_datas = json.load(f)
            ID_string = 'ID'
    elif dataset == "grailqa":
        with open("../../data/dataset/grailqa.json", "r") as f:
            ground_truth_datas = json.load(f)
            ID_string = 'qid'

    num_right = 0
    num_error = 0
    for key,data in output_datas.items():
        # if_true = False
        answers = align(dataset, ID_string, key, ground_truth_datas)
        results = data['predict_answers'] #List
        
        IF_TRUE = False
        for result in results:
            if exact_match(result, answers):
                IF_TRUE = True
                break
        if IF_TRUE:
            num_right+=1
        else:
            num_error+=1

    print("Hit*: {:.3f}".format(float(num_right/len(output_datas))))
    # print("right: {}, error: {}".format(num_right, num_error))

    


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="webqsp or grailqa or cwq")
    parser.add_argument("--input", type=str, default="the input file you want to eval,the file can be json or jsonl")#input_file should be in key:value format, where key is id and answers should be in value's predict_answers field
    args = parser.parse_args()
    
    predict_data = align_for_baseline(args.input)

    eval_containment_hit(predict_data,args.dataset) 
    eval_answer_f1_router(predict_data,args.dataset) 

            
    
    