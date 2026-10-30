import re
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import utils.predictors as predictors
import argparse
import utils.utils as utils
import utils.prompts as prompts
import utils.pairwise as pairwise
from datetime import datetime
import time
import os


def load_sentence_encoder_with_retry(model_name, max_retries=3, retry_delay_seconds=2):
    resolved_model_name = predictors.resolve_sentence_encoder_name(model_name)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return predictors.SentenceTransformer(resolved_model_name)
        except Exception as exc:
            last_error = exc
            print(f"Sentence encoder load attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(retry_delay_seconds)
    raise RuntimeError(
        f"{predictors.format_sentence_encoder_load_error(resolved_model_name, last_error)} "
        f"Unable to load the encoder after {max_retries} attempts. "
        "Pairwise KTAS retrieval requires the sentence encoder to be available."
    ) from last_error
  
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


def resolve_serialization(args):
    if args.serialization is not None:
        return args.serialization
    if args.dataset == "Triage-KTAS" and args.strategy == "KATE":
        return "commas"
    return "natural"

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
                try:
                    repaired_response = repair_json_with_model(response_text, model=model, debug=debug)
                    repaired_data = json.loads(repaired_response)
                    acuity = repaired_data.get('Acuity', acuity)
                except (json.JSONDecodeError, TypeError):
                    pass
            response_data['Acuity'] = acuity

        # Retry loop: up to 3 attempts to get valid acuity via repair
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response_data['Acuity'] = normalize_acuity_value(response_data['Acuity'])
                return response_data  # Success, return early
            except ValueError as e:
                if attempt < max_attempts - 1:
                    # Try repair mechanism
                    try:
                        print(f"Attempt {attempt + 1}/{max_attempts}: Invalid acuity, calling repair...")
                        repaired_response = repair_json_with_model(response_text, model=model, debug=debug)
                        repaired_data = json.loads(repaired_response)
                        response_data['Acuity'] = repaired_data.get('Acuity', response_data['Acuity'])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass  # Continue to next attempt
                else:
                    # All attempts failed, return False
                    print(f"Failed to extract valid acuity after {max_attempts} attempts, marking as False")
                    response_data['Acuity'] = False
                    return response_data

        return response_data

    lines = response_text.splitlines()
    last_five_lines = lines[-5:]
    shortened_text = "\n".join(last_five_lines)

    # Retry loop: up to 3 attempts to get valid acuity
    max_attempts = 3
    for attempt in range(max_attempts):
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
            if attempt < max_attempts - 1:
                print(f"Attempt {attempt + 1}/{max_attempts}: Failed to extract acuity, retrying...")
                continue
            else:
                num = heuristic_acuity_from_text(response_text)

        try:
            return {"Acuity": normalize_acuity_value(num)}
        except ValueError:
            if attempt == max_attempts - 1:
                # All attempts failed, return False
                print(f"Failed to extract valid acuity after {max_attempts} attempts, marking as False")
                return {"Acuity": False}
            else:
                print(f"Attempt {attempt + 1}/{max_attempts}: Invalid acuity {num}, retrying...")
                continue


def repair_pairwise_decision_with_model(response_text, model, debug):
    repair_prompt = (
        "Extract the final pairwise queue-ordering decision from the following model response.\n"
        "Respond with exactly one label and nothing else:\n"
        "A_BEFORE_B\n"
        "B_BEFORE_A\n\n"
        f"Response:\n\"\"\"{response_text}\"\"\""
    )
    return utils.query_llm(repair_prompt, model=model, debug=debug)


def parse_pairwise_response_with_fallback(response_text, model, debug):
    try:
        return pairwise.normalize_pairwise_decision(response_text)
    except ValueError as exc:
        print(f"Direct pairwise parse failed: {exc}")
        repaired_response = repair_pairwise_decision_with_model(response_text, model=model, debug=debug)
        return pairwise.normalize_pairwise_decision(repaired_response)
    
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
        
def parse_two_call_response(response, metadata):
    """Extract acuity from KATE2Call combined response using the Acuity_Response metadata."""
    acuity_response = metadata.get('Acuity_Response', '')
    # Split reasoning from acuity suffix
    reasoning = response.split('\n\nACUITY:')[0].strip() if '\n\nACUITY:' in response else response
    num = extract_num(acuity_response)
    if isinstance(num, str) and 'Error' in num:
        num = heuristic_acuity_from_text(acuity_response)
    try:
        acuity = normalize_acuity_value(num)
    except ValueError:
        acuity = False
    return {"Acuity": acuity, "Reasoning": reasoning}


def predict(*, index, row, predictor, model, strategy, return_json, k_shots, serialization, vitals_off, bias, debug=False, worst_example=False):
    prompt, response, metadata = predictor.predict(
        row=row,
        model=model,
        k_shots=k_shots,
        return_json=return_json,
        serialization_strategy=serialization,
        vitals_off=vitals_off,
        bias=bias,
        debug=debug,
        worst_example=worst_example)

    # KATE2Call uses two separate LLM calls — parse differently
    if strategy == 'KATE2Call':
        response_data = parse_two_call_response(response, metadata)
        if index == 0:
            return {"prompt": prompt, "Estimated_Acuity": response_data['Acuity'], "Reasoning": response_data['Reasoning'], **metadata, **row.to_dict()}
        else:
            return {"Estimated_Acuity": response_data['Acuity'], "Reasoning": response_data['Reasoning'], **metadata, **row.to_dict()}

    if return_json:
        response_data = parse_response_with_fallback(response, model=model, return_json=True, debug=debug)
        if index==0:
            return {
                "prompt": prompt,
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response_data.get('Reasoning', response),
                **metadata,
                **row.to_dict()
            }
        else:
            return {
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response_data.get('Reasoning', response),
                **metadata,
                **row.to_dict()
            }
    else:
        response_data = parse_response_with_fallback(response, model=model, return_json=False, debug=debug)
        if index==0:
            return {
                "prompt": prompt,
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response,
                **metadata,
                **row.to_dict()
            }
        else:
            return {
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response,
                **metadata,
                **row.to_dict()
            }


def predict_pairwise(*, index, row, predictor, model, strategy, k_shots, serialization, debug=False):
    prompt, response = predictor.predict(
        row=row,
        model=model,
        k_shots=k_shots,
        serialization_strategy=serialization,
        debug=debug,
    )
    decision = parse_pairwise_response_with_fallback(response, model=model, debug=debug)
    result = {
        "Predicted_Order": decision,
        **row.to_dict(),
    }
    if index == 0:
        result["prompt"] = prompt
    return result


def build_output_filepath(
    savepath,
    model,
    predictive_strategy,
    serialization,
    json_param,
    detailed_instructions,
    start_index,
    end_index,
    timestamp,
    k_shots=None,
    k_shots_ablation=False,
):
    output_filepath = f"{savepath}_{predictive_strategy}_{model}"
    if json_param:
        output_filepath = output_filepath + "_json"
    if detailed_instructions:
        output_filepath = output_filepath + "_detailed"
    if 'FewShot' in predictive_strategy or 'KATE' in predictive_strategy:
        output_filepath = output_filepath + f"_{k_shots}_shots"
    if k_shots_ablation:
        output_filepath = output_filepath + "_ablation"
    return output_filepath + f"_{serialization}_{start_index}_{end_index}_{timestamp}.csv"
        
def save_csv(df, savepath, model, predictive_strategy, serialization,json_param, detailed_instructions,  start_index, end_index, timestamp, k_shots=None, k_shots_ablation=False):
    output_filepath = build_output_filepath(
        savepath,
        model,
        predictive_strategy,
        serialization,
        json_param,
        detailed_instructions,
        start_index,
        end_index,
        timestamp,
        k_shots=k_shots,
        k_shots_ablation=k_shots_ablation,
    )
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
    parser.add_argument("--serialization", type=str, choices=["spaces", 'commas', "newline","json","natural","natural_sex_race","natural_full"], help="serialization prompt to use")
    parser.add_argument("--vitals_off", action="store_true", help="Turns on vitals off ablation prompt")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--k_shots_ablation", action="store_true", help="Change the parameters for k_shots ablation")
    parser.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434/api/chat", help="Ollama chat endpoint")
    parser.add_argument("--ollama_timeout", type=int, default=180, help="Ollama request timeout in seconds")
    parser.add_argument("--ollama_max_retries", type=int, default=3, help="Maximum Ollama retries per request")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for local models")
    parser.add_argument("--num_predict", type=int, default=400, help="Maximum generated tokens for local models")
    parser.add_argument("--save_every", type=int, default=25, help="Save checkpoint after this many new predictions")
    parser.add_argument("--pairwise_input", type=str, help="Override CSV filepath for KTAS pairwise input rows")
    parser.add_argument("--encoder_retries", type=int, default=3, help="Number of retries when loading the Hugging Face sentence encoder")
    parser.add_argument("--encoder_retry_delay", type=int, default=2, help="Seconds to wait between sentence encoder load retries")
    parser.add_argument("--run_id", type=str, help="Stable run identifier used for output/checkpoint filenames")
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from an existing checkpoint for the same run_id and arguments")
    parser.add_argument("--with_reasoning", action="store_true", help="Use train_with_reasoning.csv as the retrieval pool (includes reasoning in retrieved examples)")
    parser.add_argument("--reasoning_file", type=str, default=None, help="Specify custom reasoning CSV file path (e.g. train_with_reasoning_kate.csv)")
    parser.add_argument("--worst_example", action="store_true", help="Use the worst-ranked (least similar) retrieved example instead of the best")

    args = parser.parse_args()
    args.serialization = resolve_serialization(args)
    dataset_key = args.dataset.lower()
    if pairwise.is_pairwise_strategy(args.strategy):
        if args.dataset != "Triage-KTAS-Pairwise":
            raise ValueError("Pairwise strategies are only supported with the Triage-KTAS-Pairwise dataset.")
    else:
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
    test_filepath = args.pairwise_input or utils._DATASETS[args.dataset]['test_set_filepath']
    format = utils._DATASETS[args.dataset]['format']
    test_dataset = utils.load_dataset(test_filepath, format, args.start, args.end)
    is_pairwise = pairwise.is_pairwise_strategy(args.strategy)

    if is_pairwise and utils._DATASETS[args.dataset]["target"] in test_dataset.columns:
        equal_priority_rows = test_dataset["A_KTAS_expert"] == test_dataset["B_KTAS_expert"]
        if equal_priority_rows.any():
            raise ValueError("Equal-KTAS rows are out of scope for pairwise KTAS and must be removed from the input CSV.")

    print("Potentially loading existing predictions...")
    predictions = []
    run_timestamp = args.run_id or datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    auto_resume_name = None
    if args.auto_resume and args.run_id and not args.load_predictions:
        checkpoint_prefix = "./results/checkpoints/" + args.dataset + "/" + args.dataset
        checkpoint_filepath = build_output_filepath(
            checkpoint_prefix,
            args.model,
            args.strategy,
            args.serialization,
            args.json,
            args.detailed_instructions,
            args.start,
            args.end,
            run_timestamp,
            k_shots=args.k_shots,
            k_shots_ablation=args.k_shots_ablation,
        )
        if os.path.exists(checkpoint_filepath):
            auto_resume_name = os.path.splitext(os.path.basename(checkpoint_filepath))[0]
            print(f"Auto-resume checkpoint found: {checkpoint_filepath}")

    if args.load_predictions or auto_resume_name:
        predictions_name = args.load_predictions or auto_resume_name
        predictions = utils.load_predictions(predictions_name, format=utils._DATASETS[args.dataset]['format'], save_path="./results/checkpoints/" + args.dataset + '/')
        predictions = predictions.to_dict('records')
        num_existing_predictions = len(predictions)
        print(f"Loaded {num_existing_predictions} existing predictions.")
    else:
        num_existing_predictions = 0
    num_new_predictions_needed = (args.end - args.start) + 1 - num_existing_predictions
    
    if args.dataset == 'Triage-Counterfactual':
        args.k_shots = False
    if is_pairwise and not args.k_shots:
        raise ValueError("Pairwise KTAS retrieval requires --k_shots to be greater than 0.")
    if args.k_shots:
        if is_pairwise:
            training_df = utils.load_dataset(utils._DATASETS["Triage-KTAS"]["training_set_filepath"], format, 0, 1000000)
        elif args.k_shots_ablation:
            training_df = utils.load_dataset(utils._DATASETS[args.dataset]['full_training_set_filepath'], format, 0, 1000000)
        else:
            training_df = utils.load_dataset(utils._DATASETS[args.dataset]['training_set_filepath'], format, 0, 1000000)

        if args.reasoning_file and args.dataset.lower() == 'triage-ktas':
            reasoning_path = f"./data/KTAS/{args.reasoning_file}"
            training_df = utils.load_dataset(reasoning_path, format, 0, 1000000)
        elif args.with_reasoning and args.dataset.lower() == 'triage-ktas':
            reasoning_path = "./data/KTAS/train_with_reasoning.csv"
            training_df = utils.load_dataset(reasoning_path, format, 0, 1000000)

    print(f"Making {num_new_predictions_needed} new predictions...")
    if is_pairwise:
        complaint_embeddings = np.load(utils._DATASETS["Triage-KTAS"]["training_complaint_embeddings_filepath"], allow_pickle=True)
        diagnosis_embeddings = np.load(utils._DATASETS["Triage-KTAS"]["training_diagnosis_embeddings_filepath"], allow_pickle=True)
        encoder = load_sentence_encoder_with_retry(
            'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
            max_retries=args.encoder_retries,
            retry_delay_seconds=args.encoder_retry_delay,
        )
        retriever = pairwise.PairRetriever(
            training_set=training_df,
            complaint_embeddings=complaint_embeddings,
            diagnosis_embeddings=diagnosis_embeddings,
            encoder=encoder,
            examples_per_side=args.k_shots,
        )
        predictor = pairwise.PairPredictor(
            retriever=retriever,
            hippa=utils._DATASETS[args.dataset]['hippa'],
            strategy=args.strategy,
        )
    else:
        predictor = predictors.Predictor(args.dataset,
                                         strategy=args.strategy, 
                                         hippa=utils._DATASETS[args.dataset]['hippa'],
                                         target = utils._DATASETS[args.dataset]['target'],
                                         training_set= training_df if args.k_shots else None,
                                         k_shots_ablation=args.k_shots_ablation)

    # Initialize counter for saving progress
    new_predictions_since_last_save = 0
    total_predictions_made = num_existing_predictions

    if num_new_predictions_needed > 0:
        if utils._DATASETS[args.dataset]['format'] == 'csv':
            for i, row in tqdm(test_dataset.loc[num_existing_predictions:args.end].iterrows()):
                # Prompting and LLM Logic is all handled here
                if is_pairwise:
                    prediction = predict_pairwise(
                        index=i,
                        row=row,
                        predictor=predictor,
                        model=args.model,
                        strategy=args.strategy,
                        k_shots=args.k_shots,
                        serialization=args.serialization,
                        debug=args.debug,
                    )
                else:
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
                        debug=args.debug,
                        worst_example=args.worst_example
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
    target_column = utils._DATASETS[args.dataset]['target']
    if target_column in test_dataset.columns:
        print("Evaluating predictions...")
        try:
            ground_truths = test_dataset[target_column]
            if is_pairwise:
                predictions = predictions_df['Predicted_Order']
                metrics = utils.evaluate_pairwise_ordering_predictions(predictions, ground_truths, by_class=True)
            else:
                predictions = predictions_df['Estimated_Acuity']
                # Filter out False values (failed extractions) and convert to int
                valid_mask = predictions.apply(lambda x: isinstance(x, (int, float)) and not isinstance(x, bool))
                valid_mask = valid_mask & (predictions != False) & (predictions.notna())

                if not valid_mask.all():
                    failed_count = (~valid_mask).sum()
                    print(f"Warning: {failed_count} predictions failed extraction and will be excluded from evaluation")
                    predictions = predictions[valid_mask]
                    ground_truths = ground_truths[valid_mask]

                if len(predictions) == 0:
                    print("Skipping evaluation: no valid predictions")
                    metrics = {}
                else:
                    # Convert all predictions to int to avoid type mixing
                    predictions = predictions.astype(int)
                    metrics = utils.evaluate_predictions(predictions, ground_truths, ordinal=True, by_class=True)
            print("Overall Metrics:", metrics)
            utils.save_metrics(metrics, output_filepath.split('.csv')[0])
            print("Evaluation complete. Metrics and plots saved.")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("Skipping metrics calculation, but data has been saved.")
    else:
        print(f"Skipping evaluation because {target_column} is not present in the input file.")
    
