import re
import pandas as pd
import json
from tqdm import tqdm
from triage_utils import predictors
import argparse
from triage_utils import core as utils
from triage_utils import prompts
from datetime import datetime
import time
import os
  
def extract_num(answer_text):
    # Use regex to find all floats and integers in the answer text
    matches = re.findall(r'\d+\.\d+|\d+', answer_text)
    
    # Convert all matches to floats for consistent processing
    matches = [float(num) for num in matches]

    if len(matches) == 1:
        return matches[0]  # Return the single found number
    elif len(matches) > 1:
        return f"Error: Multiple numbers found. Please verify the data: {answer_text}"
    else:
        return "Error: No acuity number found" 

def heuristic_acuity_from_text(text):
    if text is None:
        return "Error: No acuity number found"
    ktas_match = re.search(r'"Acuity"\s*:\s*"?([1-5])"?', text)
    if not ktas_match:
        ktas_match = re.search(r"\b(?:KTAS|acuity|level)\b[^0-9]{0,10}([1-5])\b", text, flags=re.IGNORECASE)
    if not ktas_match:
        ktas_match = re.search(r"\b([1-5])\b", text)
    if ktas_match:
        return int(ktas_match.group(1))
    return "Error: No acuity number found"

def repair_json_with_model(response_text, model, debug):
    repair_prompt = (
        "Convert the following triage prediction into strict JSON with exactly one key named "
        "\"Acuity\" whose value is a single integer from 1 to 5.\n\n"
        f"Prediction:\n\"\"\"{response_text}\"\"\""
    )
    return utils.query_llm(repair_prompt, model=model, debug=debug, return_json=True)

def normalize_acuity_value(value):
    if isinstance(value, str):
        value = extract_num(value)
    if isinstance(value, float):
        value = int(value)
    if isinstance(value, int) and 1 <= value <= 5:
        return value
    raise ValueError(f"Invalid acuity value: {value}")

def parse_response_with_fallback(response_text, model, return_json, debug):
    if return_json:
        try:
            response_data = json.loads(response_text)
        except (TypeError, json.JSONDecodeError) as e:
            print("Error decoding JSON:", e)
            print("Raw response causing error:", response_text)
            repaired_response = repair_json_with_model(response_text, model=model, debug=debug)
            try:
                response_data = json.loads(repaired_response)
            except (TypeError, json.JSONDecodeError):
                acuity = heuristic_acuity_from_text(response_text)
                if isinstance(acuity, str):
                    raise ValueError(f"Unable to parse or repair model response: {response_text}")
                response_data = {"Acuity": acuity}

        if 'Acuity' not in response_data:
            acuity = heuristic_acuity_from_text(response_text)
            if isinstance(acuity, str):
                raise ValueError(f"Model response missing Acuity field: {response_text}")
            response_data['Acuity'] = acuity
        response_data['Acuity'] = normalize_acuity_value(response_data['Acuity'])
        return response_data

    lines = response_text.splitlines()
    last_five_lines = lines[-5:]
    shortened_text = "\n".join(last_five_lines)
    answer_text = utils.query_llm(
        "Extract the estimated acuity from the following information and output the number alone. "
        "If the estimate is uncertain, just choose one that is best.\n\n"
        f"\"\"\"{shortened_text}.\"\"\"",
        model=model,
        debug=debug,
    )
    num = extract_num(answer_text)
    time.sleep(1)
    if isinstance(num, str) and 'Error' in num:
        num = heuristic_acuity_from_text(response_text)
    return {"Acuity": normalize_acuity_value(num)}
    
# def create_prompt_bias(row,strategy=None, return_json=False, vitals_off=False):
#     task_description = "Acuity is assessed using the Emergency Severity Index (ESI) Five Level triage system. This priority is assigned by a registered nurse. Level 1 is the highest priority, while level 5 is the lowest priority"
#     starting_prompt = f"Here is the profile of a {row['Race']} {row['Sex']} patient" 
#     cot, reasoning = "", ""
#     if strategy=='CoT':
#         cot, reasoning = "Let's think step by step", "your step-by-step reasoning in the key 'reasoning' and "
#     if vitals_off:
#         serialization_prompt=f"""pain   chiefcomplaint
# {row['pain']}   {row['chiefcomplaint']}"""
#     else:
#         serialization_prompt=f"""temperature   heartrate   resprate   o2sat   sbp   dbp   pain   chiefcomplaint
# {row['temperature']}   {row['heartrate']}   {row['resprate']}   {row['o2sat']}   {row['sbp']}   {row['dbp']}   {row['pain']}   {row['chiefcomplaint']}"""
#     if return_json:
#         return f"""{starting_prompt}:

# {serialization_prompt}

# Estimate their acuity from 1 to 5 based on the following guidelines: {task_description}. {cot}

# Answer in valid JSON format, providing {reasoning}acuity as a single numeric value in the key 'acuity'."""
#     else:    
#         return f"""{starting_prompt}:

# {serialization_prompt}

# Estimate their acuity from 1-5 based on the following guidelines: {task_description}. {cot}
#         """
        
def predict(*, index, row, predictor, model, strategy, return_json, k_shots, serialization, vitals_off, bias, debug=False):
    prompt, response = predictor.predict(
        row=row,
        model=model,
        k_shots=k_shots,
        return_json=return_json,
        serialization_strategy=serialization,
        vitals_off=vitals_off,
        bias=bias,
        debug=debug)
    if return_json:
        response_data = parse_response_with_fallback(response, model=model, return_json=True, debug=debug)
        if index==0:
            return {
                "prompt": prompt,
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response_data['Reasoning'] if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        else:
            return {
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response_data['Reasoning'] if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
    else: 
        response_data = parse_response_with_fallback(response, model=model, return_json=False, debug=debug)
        if index==0:
            return {
                "prompt": prompt,
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        else:
            return {
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        
def save_csv(df, savepath, model, predictive_strategy, serialization,json_param, detailed_instructions,  start_index, end_index, timestamp, k_shots=None, k_shots_ablation=False):
    output_filepath = f"{savepath}_{predictive_strategy}_{model}"
    if json_param:
        output_filepath = output_filepath + "_json"
    if detailed_instructions:
        output_filepath = output_filepath + "_detailed"
    if 'FewShot' in predictive_strategy or 'KATE' in predictive_strategy:
        output_filepath = output_filepath + f"_{k_shots}_shots"
    if k_shots_ablation:
        output_filepath = output_filepath + "_ablation"
    output_filepath = output_filepath + f"_{serialization}_{start_index}_{end_index}_{timestamp}.csv"
    # Save the DataFrame to a CSV file
    df.to_csv(output_filepath, index=False)
    print(f"DataFrame saved to {output_filepath}")
    return output_filepath
    
def load_csv(savepath, model, predictive_strategy, serialization,  json_param, detailed_instructions, start_index, end_index, timestamp, k_shots =None, k_shots_ablation=False):
    input_filepath = f"{savepath}_{predictive_strategy}_{model}"
    if json_param:
        input_filepath = input_filepath + "_json"
    if detailed_instructions:
        input_filepath = input_filepath + "_detailed"
    if k_shots is not None:
        input_filepath = input_filepath + f"_{k_shots}"
    if k_shots_ablation:
        input_filepath = input_filepath + "_ablation"
    input_filepath = input_filepath + f"_{serialization}_{start_index}_{end_index}_{timestamp}.csv"
    
    try:
        df = pd.read_csv(input_filepath)
        print(f"DataFrame loaded from {input_filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found: {input_filepath}")
        return None
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make predictions on medical QA dataset using LLMs.")
    parser.add_argument("--dataset", type=str, required=True, choices=utils._DATASETS.keys(), help="Name of the dataset to evaluate")
    parser.add_argument("--start", default=0,type=int, help="Start index of the samples to evaluate")
    parser.add_argument("--end", default=1000, type=int, help="End index of the samples to evaluate")
    parser.add_argument("--model", required=True, type=str, default="gpt-4o-mini", help="LLM model to use.")
    parser.add_argument("--strategy", required=True, type=str, default="standard", help="Prediction strategy to use")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials for USC strategy")
    parser.add_argument("--k_shots", type=int, default=5, help="Number of shots for Few-Shot")
    parser.add_argument("--load_predictions", type=str, help="Timestamp & model_name & # of questions to existing predictions to load")
    parser.add_argument("--json", action="store_true", help="Turns on internal usage of json formats for the LLM API")
    parser.add_argument("--detailed_instructions", action="store_true", help="Turns on detailed instructions")
    parser.add_argument("--bias", action="store_true", help="Enables bias prompt")
    parser.add_argument("--serialization", default='natural',type=str, choices=["spaces", 'commas', "newline","json","natural","natural_sex_race","natural_full"], help="serialization prompt to use")
    parser.add_argument("--vitals_off", action="store_true", help="Turns on vitals off ablation prompt")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--k_shots_ablation", action="store_true", help="Change the parameters for k_shots ablation")
    parser.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434/api/chat", help="Ollama chat endpoint")
    parser.add_argument("--ollama_timeout", type=int, default=180, help="Ollama request timeout in seconds")
    parser.add_argument("--ollama_max_retries", type=int, default=3, help="Maximum Ollama retries per request")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for local models")
    parser.add_argument("--num_predict", type=int, default=400, help="Maximum generated tokens for local models")
    parser.add_argument("--save_every", type=int, default=25, help="Save checkpoint after this many new predictions")

    args = parser.parse_args()
    dataset_key = args.dataset.lower()
    supported_strategies = prompts.INSTRUCTIONS.get(dataset_key, {})
    if args.strategy not in supported_strategies:
        raise ValueError(f"Strategy {args.strategy} not supported for dataset {args.dataset}. Supported strategies: {', '.join(sorted(supported_strategies.keys()))}")

    utils.set_ollama_config(
        url=args.ollama_url,
        timeout=args.ollama_timeout,
        max_retries=args.ollama_max_retries,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    if args.dataset == 'Triage-KTAS' and args.strategy == 'KATE' and utils.get_model_provider(args.model) in {'ollama', 'litellm'} and not args.json:
        print("Enabling --json automatically for KTAS KATE runs on Ollama/LiteLLM models.")
        args.json = True

    print("Loading Dataset...")
    test_filepath= utils._DATASETS[args.dataset]['test_set_filepath']
    format = utils._DATASETS[args.dataset]['format']
    test_dataset = utils.load_dataset(test_filepath, format, args.start, args.end)

    print("Potentially loading existing predictions...")
    predictions = []
    if args.load_predictions:
        predictions = utils.load_predictions(args.load_predictions, format=utils._DATASETS[args.dataset]['format'], save_path="./results/checkpoints/" + args.dataset + '/')
        predictions = predictions.to_dict('records')
        num_existing_predictions = len(predictions)
        print(f"Loaded {num_existing_predictions} existing predictions.")
    else:
        num_existing_predictions = 0
    num_new_predictions_needed = (args.end - args.start) + 1 - num_existing_predictions
    
    if args.dataset == 'Triage-Counterfactual':
        args.k_shots = False
    needs_training_set = args.strategy in {"FewShot", "FewShotCoT", "KATE", "KATECoT", "KATEAutoCoT"}
    training_df = None
    if needs_training_set:
        training_filepath_key = 'training_set_filepath'
        if args.dataset == 'Triage-ESI-Cleaned' and args.strategy in {"KATE", "KATECoT", "KATEAutoCoT"}:
            training_filepath_key = 'kate_training_set_filepath'
        if args.k_shots_ablation:
            training_df = utils.load_dataset(utils._DATASETS[args.dataset]['full_training_set_filepath'], format, 0, 1000000)
        else:
            training_df = utils.load_dataset(utils._DATASETS[args.dataset][training_filepath_key], format, 0, 1000000)

    print(f"Making {num_new_predictions_needed} new predictions...")
    predictor = predictors.Predictor(args.dataset,
                                     strategy=args.strategy, 
                                     hippa=utils._DATASETS[args.dataset]['hippa'],
                                     target = utils._DATASETS[args.dataset]['target'],
                                     training_set=training_df,
                                     k_shots_ablation=args.k_shots_ablation)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Initialize counter for saving progress
    new_predictions_since_last_save = 0
    total_predictions_made = num_existing_predictions

    if num_new_predictions_needed > 0:
        if utils._DATASETS[args.dataset]['format'] == 'csv':
            for i, row in tqdm(test_dataset.loc[num_existing_predictions:args.end].iterrows()):
                # Prompting and LLM Logic is all handled here
                prediction = predict(
                    index=i,
                    row=row,
                    predictor=predictor,
                    model=args.model,
                    strategy=args.strategy,
                    return_json=args.json,
                    k_shots=args.k_shots,
                    serialization=args.serialization,
                    vitals_off=args.vitals_off,
                    bias=args.bias,
                    debug=args.debug
                )
                predictions.append(prediction)
                new_predictions_since_last_save += 1
                total_predictions_made += 1

                if new_predictions_since_last_save >= args.save_every:
                    # Save predictions to disk
                    predictions_df = pd.DataFrame(predictions)
                    save_path = "./results/checkpoints/" + args.dataset + '/'
                    os.makedirs(save_path, exist_ok=True)
                    save_path = save_path + args.dataset
                    save_csv(predictions_df, save_path, 
                             args.model, 
                             args.strategy, 
                             args.serialization,
                             args.json, 
                             args.detailed_instructions,
                             args.start,
                             args.end,
                             run_timestamp, k_shots=args.k_shots)
                    new_predictions_since_last_save = 0
                    print(f"Saved progress after {len(predictions)} predictions.")
            predictions_df = pd.DataFrame(predictions)
    else:
        print("No new predictions needed.")

    # Save combined predictions one last time
    save_path = "./results/" + args.dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    save_path = save_path + args.dataset
    output_filepath = save_csv(predictions_df, save_path, 
                     args.model, 
                     args.strategy, 
                     args.serialization,
                     args.json, 
                     args.detailed_instructions,
                     args.start,
                     args.end,
                    run_timestamp,k_shots=args.k_shots,
                    k_shots_ablation=args.k_shots_ablation)

    print("Processing complete. Predictions saved.")
    
    # Evaluate predictions
    print("Evaluating predictions...")
    target_col = utils._DATASETS[args.dataset]['target']
    ground_truths = pd.to_numeric(test_dataset[target_col], errors='coerce').astype(int)
    predictions = pd.to_numeric(predictions_df['Estimated_Acuity'], errors='coerce').astype(int)

    metrics = utils.evaluate_predictions(predictions, ground_truths, ordinal=True, by_class=True)
    print("Overall Metrics:", metrics)
    utils.save_metrics(metrics, output_filepath.split('.csv')[0])
    print("Evaluation complete. Metrics and plots saved.")
    
