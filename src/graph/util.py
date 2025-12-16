import json
import time
from openai import OpenAI
import openai
import random
from config import OPENAI_API_KEY, QWEN_API_KEYS,QWEN_BASE_URL,OPENAI_BASE_URL
from prompts import KGQA_FEW_SHOT_PROMPT
def process_dataset_for_kBET(data,dataset_name):
    if dataset_name == "cwq":
        return data
    elif dataset_name == "webqsp":
        processed_data = []
        for item in data:
            item_to_write = {}
            item_to_write = item.copy()
            item_to_write["ID"] = item["QuestionId"]
            item_to_write["question"] = item["ProcessedQuestion"]
            item_to_write["answer"] = item["Parses"][0]["Answers"]
            processed_data.append(item_to_write)
        return processed_data
    elif dataset_name == "grailqa":
        process_data = []
        for item in data:
            item_to_write = {}
            item_to_write = item.copy()
            item_to_write["ID"] = item["qid"]
            item_to_write["question"] = item["question"]
            process_data.append(item_to_write)
        
        return process_data
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}, supported types: cwq/webqsp/grailqa")



def check_string(string):
    return "{" in string

def clean_results(string):
    if "{" in string:
        start = string.rfind("{") + 1
        end = string.rfind("}")
        response = string[start:end]
        return response
    else:
        return "NULL"

def run_llm_inference(prompt, temperature, max_tokens, engine="qwen-plus"):
    print("LLM engine:",engine)
    if "qwen" in engine or "deepseek" in engine:
        apikey = random.choice(QWEN_API_KEYS)
        client = OpenAI(
            api_key=apikey,
            base_url=QWEN_BASE_URL
        )
    elif "gpt" in engine :
        key = random.choice(OPENAI_API_KEY)
        client = OpenAI(api_key=key,base_url=OPENAI_BASE_URL)


    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if engine == "qwen-plus":
                response = client.chat.completions.create(
                        model=engine,
                        messages = messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed = 0,
                        extra_body={"enable_thinking": False}
                        )
            else:
                response = client.chat.completions.create(
                            model=engine,
                            messages = messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            )
            result = response.choices[0].message.content

            return result
            
        except openai.RateLimitError:
                print(f"Rate limit exceeded, retry {retry_count+1}/{max_retries}")
        except Exception as e:
            print(f"Unexpected error: {e}, retry {retry_count+1}/{max_retries}")
        
        retry_count += 1
        time.sleep(2)
        
    print(f"Failed after {max_retries} retries")    
    return ""


def perform_reasoning(question,cluster_chain_of_entities,model,temperature,max_tokens):
    """
    Function to perform direct reasoning, taking a list of triples and a model as input to answer the question.
    """
    prompt = KGQA_FEW_SHOT_PROMPT % (str(cluster_chain_of_entities), question)
    response = run_llm_inference(prompt, temperature, max_tokens, model)

    return response

if __name__ == "__main__":
    result = run_llm_inference("hello",0,250,"qwen-plus")
    print(result)
    