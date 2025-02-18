import requests
import json
from tqdm import tqdm
from dotenv import load_dotenv
import os
import re
from typing import List
from openai import OpenAI
from datetime import datetime
from time import sleep
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_payload(model_name, system_prompt, user_prompt):
    return json.dumps(
        {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def extract_result_sentences(text):
    matches = re.findall(r"<result>(.*?)</result>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parsing error: {e}")
                    continue
    return data


class Agent:
    def __init__(self, Skey, Baseurl="https://api.claudeshop.top"):
        self.Baseurl = Baseurl
        self.Skey = Skey
        self.url = Baseurl + "/v1/chat/completions"
        self.headers = {"Accept": "application/json", "Authorization": f"Bearer {Skey}", "User-Agent": "Apifox/1.0.0 (https://apifox.com)", "Content-Type": "application/json"}

    def __call__(self, model_name, system_prompt, user_prompt):
        while True:
            try:
                payload = create_payload(model_name, system_prompt, user_prompt)
                response = requests.post(self.url, headers=self.headers, data=payload)
                response_data = response.json()
                response.raise_for_status()  
                content = response_data["choices"][0]["message"]["content"]
                return content
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                sleep(1)  # Wait for 1 second before retrying
            except KeyError as e:
                print(f"Failed to parse response: {e}")
                sleep(1)  # Wait for 1 second before retrying
            except Exception as e:
                print(f"An unknown error occurred: {e}")
                sleep(1)  # Wait for 1 second before retrying


class OpenAILLM:
    def __init__(self, model_name, api_key_file_path):

        self.model_name = model_name
        load_dotenv(os.path.expanduser(api_key_file_path))

        self.client = OpenAI()
        self.client.base_url = os.getenv("OPENAI_API_BASE")
        keys = os.getenv("OPENAI_API_KEY")
        self.client.api_key = keys.split(",")[0]

    def __call__(
        self,
        prompts: List[str],
        system_prompt=None,
        temperature=0.8,
        top_p=1.0,
        max_completion_tokens=4096,
        seed=42,
        max_num_retries=2,
        return_full=False,
        stop: List[str] = [],
    ) -> str:
        prompt_list = prompts
        if isinstance(prompts, str):
            prompt_list = [prompts]
        outputs = [self.generate(prompt, system_prompt, temperature, top_p, max_completion_tokens, seed, max_num_retries, return_full, stop) for prompt in prompt_list]
        if isinstance(prompts, str):
            return outputs[0]
        return outputs

    def generate(
        self,
        prompt,
        system_prompt=None,
        temperature=1,
        top_p=1.0,
        max_completion_tokens=4096,
        seed=42,
        max_num_retries=2,
        return_full=False,
        stop: List[str] = [],
    ) -> str:
        if system_prompt is not None:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

        retry = 0
        while retry < max_num_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # temperature=temperature,
                    top_p=top_p,
                    # max_completion_tokens=max_completion_tokens,
                    seed=seed,
                    stop=stop,
                )
                content = completion.choices[0].message.content
                if not return_full:
                    return content

                ret_dict = {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "model_name": self.model_name,
                    "temperature": temperature,
                    "max_tokens": max_completion_tokens,
                    "response": content,
                    "response_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    # "completion_obj": completion,
                }
                return ret_dict

            except Exception as e:
                retry += 1
                sleep(5)
                print(f"Error: {e}", flush=True)

        raise RuntimeError("Calling OpenAI failed after retrying for " f"{retry} times.")


class DeepSeek_V3:
    def __init__(self, api_key, base_url="https://api.deepseek.com", model_name="deepseek-chat"):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def __call__(self, prompts: List[str], system_prompt, temperature=0, stream=False) -> str:
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = prompts

        outputs = [self.generate(prompt, system_prompt, temperature, stream) for prompt in prompt_list]

        if isinstance(prompts, str):
            return outputs[0]
        return outputs

    def generate(self, prompt: str, system_prompt: str, temperature: float, stream: bool) -> str:
        max_retries = 50
        retry_delay = 100 
        for attempt in range(max_retries):
            try:
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                )
                print(response)
                print(response.choices[0].message.content)
                return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    sleep(retry_delay)
                else:
                    print(f"API call failed after {max_retries} attempts, returning empty string")
                    return ""


class DeepSeek_R1:
    def __init__(self, api_key, base_url="https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(self, prompts, model_name="deepseek-reasoner"):
        messages = [{"role": "user", "content": prompts}]
        response = self.client.chat.completions.create(model=model_name, messages=messages)
        # reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return content

class QwQLLM:
    def __init__(self, model_name="Qwen/QwQ-32B-Preview"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.system_message = {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."}

    def __call__(self, prompts):
        messages = [self.system_message, {"role": "user", "content": prompts}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=8192)
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


def cosine_similarity_text(str1, str2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([str1, str2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def levenshtein_similarity(str1, str2):
    dist = levenshtein_distance(str1.lower(), str2.lower())
    max_len = max(len(str1), len(str2))
    return 1 - dist / max_len


def compare_strings(str1: str, str2: str, compare_numbers: bool = True) -> bool:
    str1 = str1.replace(" ", "")
    str2 = str2.replace(" ", "")
    if len(str1) != len(str2):
        return False

    if compare_numbers:
        return str1.lower() == str2.lower()
    else:
        for i in range(len(str1)):
            if str1[i].isdigit() and str2[i].isdigit():
                continue
            if str1[i].lower() != str2[i].lower():
                return False
        return True
