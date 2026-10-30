import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

try:
    from utils.config import OPENAI_API_KEY, TOGETHER_API_KEY, DATABRICKS_TOKEN, ANTHROPIC_KEY, GEMINI_KEY
except ImportError:
    try:
        from utils.config_demo import OPENAI_API_KEY, TOGETHER_API_KEY, DATABRICKS_TOKEN, ANTHROPIC_KEY, GEMINI_KEY
    except ImportError:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
        TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
        DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
        ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY", "")
        GEMINI_KEY = os.environ.get("GEMINI_KEY", "")
LITELLM_API_KEY = os.environ.get("LITELLM", "")
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "")
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, 
                             classification_report, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import time 
import re
import json 
import urllib.error
import urllib.request
from pydantic import BaseModel

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY')) if OpenAI and os.environ.get('OPENAI_API_KEY') else None

client_safe = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-4750903324350629.9.azuredatabricks.net/serving-endpoints"
) if OpenAI and DATABRICKS_TOKEN else None

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if anthropic and ANTHROPIC_KEY else None

gemini_client = genai.Client(api_key=GEMINI_KEY) if genai and GEMINI_KEY else None

litellm_client = OpenAI(
    api_key=LITELLM_API_KEY,
    base_url=LITELLM_BASE_URL,
) if OpenAI and LITELLM_API_KEY and LITELLM_BASE_URL else None

OLLAMA_CONFIG = {
    "url": "http://127.0.0.1:11434/api/chat",
    "timeout": 180,
    "max_retries": 3,
    "temperature": 0.1,
    "num_predict": 400,
}

###################
## Data Processing
###################

# This is for gemini to return a json object with the acuity level
class Acuity(BaseModel):
    Acuity: int

# We store unanimously required & unique characteristics of the datasets here. Otherwise, edge cases will be handled elsewhere.
_DATASETS = {
    "MedQA": 
        {'filepath': "./data/medqa/questions/US/4_options/phrases_no_exclude_test.jsonl",
        'format': 'jsonl',
        'hippa': False,
        },
    "Triage-Counterfactual": 
        {'test_set_filepath': "./data/mimic-iv-public/triage_counterfactual.csv",
        'format': 'csv',
        'target': 'acuity',
        'hippa': False,
        'training_set_filepath':'./data/mimic-iv-public/triage_public.csv',
        },
    "Triage-MIMIC": 
        {'test_set_filepath': "./data/mimic-iv-private/anchor_year_group_datasets/2017_-_2019/test_dataset.csv",
        'training_set_filepath':'./data/mimic-iv-private/anchor_year_group_datasets/2014_-_2016/small_train_dataset.csv',
        'full_training_set_filepath':'./data/mimic-iv-private/anchor_year_group_datasets/2014_-_2016/train_dataset.csv',
        'full_training_embeddings_filepath':'./data/mimic-iv-private/anchor_year_group_datasets/2014_-_2016/train_full_chiefcomplaint_embeddings.npy',
        'training_embeddings_filepath':'./data/mimic-iv-private/anchor_year_group_datasets/2014_-_2016/train_chiefcomplaint_embeddings.npy',
        'format': 'csv',
        'target': 'acuity',
        'hippa': True,
        },
    "Triage-KTAS": 
        {'test_set_filepath': "./data/KTAS/test.csv",
        'training_set_filepath':'./data/KTAS/train.csv',
        'training_complaint_embeddings_filepath':'./data/KTAS/KTAS_train_chiefcomplaint_embeddings.npy',
        'training_diagnosis_embeddings_filepath':'./data/KTAS/KTAS_train_diagnosis_embeddings.npy',
        'format': 'csv',
        'target': 'KTAS_expert',
        'hippa': False,
        },
    "Triage-Handbook": 
        {'test_set_filepath': "./data/ESI-Handbook/train.csv",
        'training_set_filepath':'./data/ESI-Handbook/test.csv',
        'training_embeddings_filepath':'./data/ESI-Handbook/train_embeddings.npy',
        'test_embeddings_filepath':'./data/ESI-Handbook/test_embeddings.npy',
        'format': 'csv',
        'target': 'acuity',
        'hippa': False,
        },
    "Triage-ESI-Cleaned":
        {'test_set_filepath': "./Results/5v_relevant_features_test_500.csv",
        'training_set_filepath':'./Results/5v_relevant_features_train.csv',
        'kate_training_set_filepath':'./Results/5v_relevant_features_kate_train_50000.csv',
        'training_embeddings_filepath':'./Results/5v_relevant_features_kate_train_50000_embeddings.npy',
        'format': 'csv',
        'target': 'esi',
        'hippa': False,
        },
    ### Legacy datasets
    "Triage-Public": 
        {'filepath': "./data/mimic-iv-public/triage_public.csv",
         'format': 'csv',
         'target': 'acuity',
         'training_set_filepath':'./data/mimic-iv-public/triage_public.csv',
         'hippa': False,
        },
    "Triage-Private-Stratified": 
        {'filepath': "./data/mimic-iv-private/triage_stratified_2500.csv",
         'training_set_filepath':'./data/mimic-iv-private/triage_stratified_training.csv',
        'format': 'csv',
        'target': 'acuity',
        'hippa': True,
        },
    "Triage-Private-Sex-Race": 
        {'filepath': "./data/mimic-iv-private/full_triage_2500.csv",
         'training_set_filepath':'./data/mimic-iv-private/triage_stratified_training.csv',
        'format': 'csv',
        'target': 'acuity',
        'hippa': True,
        },
    }

def save_metrics(metrics,  filepath):
    output_file = f"{filepath}_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
             
def load_dataset(filepath, format, start_index, end_index):
    if not filepath:
        raise ValueError("Dataset not found in _DATASETS.")
    if format == 'jsonl':
        data = load_jsonl(filepath, start_index, end_index)
    elif format == 'csv':

        data = pd.read_csv(filepath).loc[start_index:end_index]

    else:
        raise ValueError(f"Unsupported format: {format}")
    return data
    
def load_jsonl(filepath, start_index, end_index):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for i, line in enumerate(f) if i <= end_index and i >= start_index]
    return data

def load_predictions(filename, format='txt', save_path="./results/"):
    if format == 'csv':
        predictions_file = os.path.join(save_path, f"{filename}.csv")
        predictions = pd.read_csv(predictions_file)
    else: 
        predictions_file = os.path.join(save_path, f"{filename}.txt")
        with open(predictions_file, 'r') as f:
            predictions = [json.loads(line.strip()) for line in f]
    return predictions

def stratified_df(df, target_col, test_size, seed=0):
    # Define stratified shuffle split
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    
    # Get the stratified sample indices
    for train_indices, test_indices in stratified_split.split(df, df[target_col]):
        stratified_train_df = df.iloc[train_indices]
        stratified_test_df = df.iloc[test_indices]
    
    return stratified_train_df.reset_index(drop=True), stratified_test_df.reset_index(drop=True)
        
###################
## Evaluation
###################

def evaluate_predictions(predicted_answers, correct_answers, ordinal=False,flexibility=1, by_class=False):
    """
    Evaluate predictions with standard metrics, Quadratic Weighted Kappa, and optionally per-class analysis.

    Args:
        predicted_answers (list): List of predicted class labels.
        correct_answers (list): List of true class labels.
        flexibility (int): Tolerance for flexibility in metrics (e.g., ±1).
        by_class (bool): Whether to compute metrics by class.

    Returns:
        dict: A dictionary containing overall and optionally per-class evaluation metrics.
    """
    # Standard metrics
    accuracy = accuracy_score(correct_answers, predicted_answers)
    precision = precision_score(correct_answers, predicted_answers, average='weighted')
    recall = recall_score(correct_answers, predicted_answers, average='weighted')
    f1 = f1_score(correct_answers, predicted_answers, average='weighted')
    
    # Adjust predictions for flexibility
    adjusted_predictions = [
        true if abs(true - pred) <= flexibility else pred
        for true, pred in zip(correct_answers, predicted_answers)
    ]

    adjusted_accuracy = accuracy_score(correct_answers, adjusted_predictions)
    adjusted_precision = precision_score(correct_answers, adjusted_predictions, average='weighted')
    adjusted_recall = recall_score(correct_answers, adjusted_predictions, average='weighted')
    adjusted_f1 = f1_score(correct_answers, adjusted_predictions, average='weighted')

    # Error metrics (MAE and MSE)
    mae = mean_absolute_error(correct_answers, predicted_answers)
    mse = mean_squared_error(correct_answers, predicted_answers)
    # Metrics by class
    if by_class:
        report = classification_report(correct_answers, predicted_answers, output_dict=True)

    # Consolidate results
    results = {
        "overall": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "adjusted_accuracy": adjusted_accuracy,
            "adjusted_precision": adjusted_precision,
            "adjusted_recall": adjusted_recall,
            "adjusted_f1": adjusted_f1,
            "mae": mae,
            "mse": mse
        }
    }
    if ordinal:
        qwk = cohen_kappa_score(correct_answers, predicted_answers, weights="quadratic")
        results['overall']["quadratic_kappa"] = qwk
    if by_class:
        results["by_class"] = report

    return results

###################
## API Calling
###################

def set_ollama_config(*, url=None, timeout=None, max_retries=None, temperature=None, num_predict=None):
    if url is not None:
        OLLAMA_CONFIG["url"] = url
    if timeout is not None:
        OLLAMA_CONFIG["timeout"] = timeout
    if max_retries is not None:
        OLLAMA_CONFIG["max_retries"] = max_retries
    if temperature is not None:
        OLLAMA_CONFIG["temperature"] = temperature
    if num_predict is not None:
        OLLAMA_CONFIG["num_predict"] = num_predict

def get_model_provider(model):
    model_lower = model.lower()
    if model_lower.startswith("claude"):
        return "claude"
    if model_lower.startswith("gemini"):
        return "gemini"
    if model_lower.startswith(("gpt", "o3", "openai-")):
        return "openai"
    if model_lower in {"azure-gpt-4.4"} or model_lower.startswith(("azure-", "azure/")):
        return "litellm"
    return "ollama"

def is_ollama_model(model):
    return get_model_provider(model) == "ollama"

def _format_messages(prompt, is_prompt_full=False):
    if is_prompt_full:
        return [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
    return [{"role": "user", "content": prompt}]

def _coerce_ollama_content(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            coerced = _coerce_ollama_content(item)
            if coerced:
                parts.append(coerced)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "response", "thinking"):
            nested = value.get(key)
            coerced = _coerce_ollama_content(nested)
            if coerced:
                return coerced
        try:
            return json.dumps(value)
        except TypeError:
            return str(value)
    return ""

def _extract_ollama_text(response_json):
    message = response_json.get("message", {})
    if isinstance(message, dict):
        content = _coerce_ollama_content(message.get("content"))
        if content:
            return content
        message_text = _coerce_ollama_content(message)
        if message_text:
            return message_text
    response = response_json.get("response")
    response_text = _coerce_ollama_content(response)
    if response_text:
        return response_text
    try:
        return json.dumps(response_json)
    except TypeError:
        pass
    raise ValueError(f"Ollama response missing readable content. Top-level keys: {sorted(response_json.keys())}")

def query_ollama(prompt, model, temperature=0, max_tokens=1000, return_json=False, is_prompt_full=False, debug=False):
    payload = {
        "model": model,
        "messages": _format_messages(prompt, is_prompt_full=is_prompt_full),
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else OLLAMA_CONFIG["temperature"],
            "num_predict": max_tokens if max_tokens is not None else OLLAMA_CONFIG["num_predict"],
        },
    }
    if return_json:
        payload["format"] = "json"
    if debug:
        print(payload)

    request = urllib.request.Request(
        OLLAMA_CONFIG["url"],
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=OLLAMA_CONFIG["timeout"]) as response:
        body = response.read().decode("utf-8")
    return _extract_ollama_text(json.loads(body))

def query_gpt_safe(prompt, model="openai-gpt-4o-high-quota-chat", return_json=False, temperature=0.0, max_tokens=1000, debug=False, tries=0, is_prompt_full=False):
    if client_safe is None:
        raise RuntimeError("Databricks/OpenAI client is unavailable. Install `openai` and configure `DATABRICKS_TOKEN`.")
    time.sleep(0.5)
    if debug:
        print(prompt)
    try:
        if is_prompt_full:
            # Format chat prompt with system and user messages
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        if 'o3' in model:
            if return_json:
                response = client_safe.chat.completions.create(
                    model=model,
                    reasoning_effort="medium",
                    messages=messages,
                    response_format={"type": "json_object"}
                )
            else:
                response = client_safe.chat.completions.create(
                    model=model,
                    reasoning_effort="medium",
                    messages=messages,
                )
        elif return_json:
            response = client_safe.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
        else:
            response = client_safe.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        response = response.choices[0].message.content.strip()
        if debug:
            print(response)
        return response
    except Exception as e:
        if tries >= 1:
            print(f"Error in query_gpt_safe after {tries + 1} tries: {e}")
            return None
            
        print("Error in query_gpt_safe. Waiting before retrying...")
        time.sleep(10)  # Wait 10 seconds before retrying
        return query_gpt_safe(prompt, model, return_json, temperature, max_tokens, debug, tries + 1, is_prompt_full)

def query_claude(prompt: str | dict, model: str, temperature: float, max_tokens: int, is_prompt_full: bool = False):
    if claude_client is None:
        raise RuntimeError("Anthropic client is unavailable. Install `anthropic` and configure `ANTHROPIC_KEY`.")
    try:
        messages = _format_messages(prompt, is_prompt_full=is_prompt_full)

        response = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return None

def query_gemini(message, model, temperature=0, max_tokens=1000):
    if gemini_client is None or types is None:
        raise RuntimeError("Gemini client is unavailable. Install `google-genai` and configure `GEMINI_KEY`.")
    response = gemini_client.models.generate_content(
        model=model,
        contents=message,
        config=types.GenerateContentConfig(
            temperature=temperature,
            seed=0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=Acuity,
        ),
    )

    return response.text

def query_openai_compatible(client_instance, prompt: str | dict, model: str, temperature: float = 0, top_p: float = 0, logprobs: bool = False, return_json: bool = False, is_prompt_full: bool = False):
    messages = _format_messages(prompt, is_prompt_full=is_prompt_full)

    if return_json:
        response = client_instance.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
    elif logprobs:
        response = client_instance.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            top_logprobs=3,
            seed=0
        )
    else:
        response = client_instance.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=0
        )
    return response

def query_gpt(prompt: str | dict, model: str = 'gpt-4o-mini', temperature: float = 0, top_p: float = 0, logprobs: bool = False, return_json: bool = False, is_prompt_full: bool = False):
    if client is None:
        raise RuntimeError("OpenAI client is unavailable. Install `openai` and configure `OPENAI_API_KEY`.")
    return query_openai_compatible(client, prompt, model, temperature, top_p, logprobs, return_json, is_prompt_full)

def query_litellm(prompt: str | dict, model: str, temperature: float = 0, top_p: float = 0, logprobs: bool = False, return_json: bool = False, is_prompt_full: bool = False):
    if litellm_client is None:
        raise RuntimeError("LiteLLM client is unavailable. Ensure `.env` contains `LITELLM` and `LITELLM_BASE_URL`.")
    return query_openai_compatible(litellm_client, prompt, model, temperature, top_p, logprobs, return_json, is_prompt_full)

def query_llm(prompt, max_tokens=1000, temperature=0, top_p=0, max_try_num=10, model="gpt-4o-mini", debug=False, return_json=False, logprobs=False):
    if debug:
        print(prompt)
        print(f"Model: {model}")
    provider = get_model_provider(model)
    if provider == "ollama":
        max_try_num = OLLAMA_CONFIG["max_retries"]
    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            if provider == "openai":
                response = query_gpt(prompt, model=model, temperature=temperature, top_p=top_p, logprobs=logprobs, return_json=return_json)
                if debug:   
                    print(response)
                if hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
                else:
                    return response.choices[0].message
            elif provider == "litellm":
                response = query_litellm(prompt, model=model, temperature=temperature, top_p=top_p, logprobs=logprobs, return_json=return_json)
                if debug:
                    print(response)
                if hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
                else:
                    return response.choices[0].message
            elif provider == "claude":
                response = query_claude(prompt, model, temperature, max_tokens)
                if return_json:
                    return re.sub(r'(?<!\\)\n', '', response)
                return response
            elif provider == "gemini":
                return query_gemini(prompt, model, temperature, max_tokens)
            elif provider == "ollama":
                return query_ollama(prompt, model, temperature=temperature, max_tokens=max_tokens, return_json=return_json, debug=debug)
            if debug:
                print(response.choices[0].message.content.strip())
            if logprobs:
                return response.choices[0].message.content.strip(), response.choices[0].logprobs
            return response.choices[0].message.content.strip()
        except Exception as e:
            if provider in {"openai", "litellm"}:
                print(f"Error making OpenAI API call: {e}")
            else: 
                print(f"Error making API call: {e}")
            curr_try_num += 1
            time.sleep(10)
            
    return None

def query_llm_full(prompt, max_tokens=1000, temperature=0, top_p=0, max_try_num=10, model="gpt-4o-mini", debug=False, return_json=False, logprobs=False):
    if debug:
        print(f"System prompt: {prompt['system']}")
        print(f"User prompt: {prompt['user']}")
        print(f"Model: {model}")
    provider = get_model_provider(model)
    if provider == "ollama":
        max_try_num = OLLAMA_CONFIG["max_retries"]
    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            if provider == "openai":
                response = query_gpt(prompt, model, temperature, top_p, logprobs, return_json, is_prompt_full=True)
            elif provider == "litellm":
                response = query_litellm(prompt, model, temperature, top_p, logprobs, return_json, is_prompt_full=True)
            elif provider == "claude":
                response = query_claude(prompt, model, temperature, max_tokens, is_prompt_full=True)
                if return_json:
                    return re.sub(r'(?<!\\)\n', '', response)
                return response
            elif provider == "gemini":
                full_prompt = f"{prompt['system']}\n\n{prompt['user']}"
                return query_gemini(full_prompt, model, temperature, max_tokens)
            elif provider == "ollama":
                return query_ollama(prompt, model, temperature=temperature, max_tokens=max_tokens, return_json=return_json, is_prompt_full=True, debug=debug)
            if debug:
                print(response.choices[0].message.content.strip())
            if logprobs:
                return response.choices[0].message.content.strip(), response.choices[0].logprobs
            return response.choices[0].message.content.strip()
        except Exception as e:
            if provider in {"openai", "litellm"}:
                print(f"Error making OpenAI API call: {e}")
            else: 
                print(f"Error making API call: {e}")
            curr_try_num += 1
            time.sleep(10)
    return None
