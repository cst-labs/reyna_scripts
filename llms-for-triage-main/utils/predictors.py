import numpy as np
import utils.utils as utils
import utils.prompts as prompts
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import os


def resolve_sentence_encoder_name(default_model_name):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_model_dir = os.path.join(
        repo_root,
        "models",
        default_model_name.split("/")[-1],
    )
    return (
        os.environ.get("TRIAGE_SENTENCE_ENCODER")
        or os.environ.get("SENTENCE_TRANSFORMER_MODEL")
        or (local_model_dir if os.path.isdir(local_model_dir) else None)
        or default_model_name
    )


def format_sentence_encoder_load_error(model_name, exc):
    message = [f"Unable to load sentence encoder '{model_name}': {exc}"]
    if "CERTIFICATE_VERIFY_FAILED" in str(exc):
        message.append(
            "This usually means HTTPS interception or a custom corporate CA is in the trust chain."
        )
        message.append(
            "Set REQUESTS_CA_BUNDLE/SSL_CERT_FILE to your organization CA bundle, or point "
            "TRIAGE_SENTENCE_ENCODER to a local Hugging Face model directory."
        )
    return " ".join(message)


class KTASSingleCaseRetriever:
    def __init__(
        self,
        *,
        training_set,
        complaint_embeddings,
        diagnosis_embeddings,
        symptom_encoder,
    ):
        self.training_set = training_set.reset_index(drop=True)
        self.complaint_embeddings = np.asarray(complaint_embeddings)
        self.diagnosis_embeddings = np.asarray(diagnosis_embeddings)
        self.embeddings_cache = np.concatenate(
            [self.complaint_embeddings, self.diagnosis_embeddings],
            axis=1,
        )
        self.symptom_encoder = symptom_encoder

    @staticmethod
    def get_top_k_similar(embedding, embeddings, k, reverse=False):
        similarities = np.dot(embeddings, embedding)
        if reverse:
            top_k_indices = np.argsort(similarities)[:k]
        else:
            top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices, similarities[top_k_indices]

    def retrieve_top_k_examples(self, row, k, worst_embedding=False):
        symptom = row["Chief_complain"]
        diagnosis = row["Diagnosis in ED"]

        if pd.isna(symptom) or symptom == "":
            combined_embedding = self.symptom_encoder.encode(diagnosis)
            top_k_indices, _ = self.get_top_k_similar(
                combined_embedding,
                self.diagnosis_embeddings,
                k * 3,
                reverse=worst_embedding,
            )
        elif pd.isna(diagnosis) or diagnosis == "":
            combined_embedding = self.symptom_encoder.encode(symptom)
            top_k_indices, _ = self.get_top_k_similar(
                combined_embedding,
                self.complaint_embeddings,
                k * 3,
                reverse=worst_embedding,
            )
        else:
            symptom_embedding = self.symptom_encoder.encode(symptom)
            diagnosis_embedding = self.symptom_encoder.encode(diagnosis)
            combined_embedding = np.concatenate([symptom_embedding, diagnosis_embedding])
            top_k_indices, _ = self.get_top_k_similar(
                combined_embedding,
                self.embeddings_cache,
                k * 3,
                reverse=worst_embedding,
            )

        vital_sign_columns = ["NRS_pain", "SBP", "DBP", "HR", "RR", "BT"]
        numeric_vital_signs = [
            col for col in vital_sign_columns if pd.api.types.is_numeric_dtype(row[col])
        ]

        query_vital_signs = row[numeric_vital_signs].values.astype(float)
        top_candidates = self.training_set.loc[top_k_indices, numeric_vital_signs]
        top_candidates = top_candidates.apply(pd.to_numeric, errors="coerce")
        top_candidates = top_candidates.dropna()
        top_candidates_values = top_candidates.values.astype(float)

        query_vital_signs_normalized = query_vital_signs / np.linalg.norm(query_vital_signs)
        top_candidates_normalized = top_candidates_values / np.linalg.norm(
            top_candidates_values,
            axis=1,
            keepdims=True,
        )

        vital_signs_similarity = np.dot(
            top_candidates_normalized,
            query_vital_signs_normalized,
        )
        if worst_embedding:
            top_k_vital_indices = np.argsort(vital_signs_similarity)[:k]
        else:
            top_k_vital_indices = np.argsort(vital_signs_similarity)[-k:][::-1]
        return self.training_set.loc[top_k_indices[top_k_vital_indices]]

class Predictor:
    """
    The predictor class is responsible for handling different prediction strategies, including:
    0) appropriate prompt for the strategy
    1) task-specific serialization of each sample 
    2) the task-specific instructions
    3) strategy logic for each prediction
    """
    def __init__(self, dataset, strategy="Vanilla", hippa=False, target=None, training_set=None, k_shots_ablation=False):
        """
        :param model: The LLM to use for predictions.
        :param strategy: Prediction strategy ("FewShot", "CoT", "Vanilla", "SelfConsistency").
        :param debug: Whether to enable debug mode.
        :param training_set: The training dataset for few-shot prompting.
        """
        self.dataset = dataset
        self.strategy = strategy
        self.target = target
        self.hippa = hippa
        self.training_set = training_set
        if training_set is not None:
            model_name = resolve_sentence_encoder_name(
                'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
            )
            try:
                self.symptom_encoder = SentenceTransformer(model_name)
            except Exception as exc:
                raise RuntimeError(format_sentence_encoder_load_error(model_name, exc)) from exc
            if self.dataset.lower() == 'triage-ktas':
                complaint_embeddings = np.load(utils._DATASETS[dataset]['training_complaint_embeddings_filepath'], allow_pickle=True)
                diagnosis_embeddings = np.load(utils._DATASETS[dataset]['training_diagnosis_embeddings_filepath'], allow_pickle=True)
                if len(self.training_set) != complaint_embeddings.shape[0]:
                    print(
                        f"Training set has {len(self.training_set)} rows but cached KTAS embeddings have "
                        f"{complaint_embeddings.shape[0]} rows; recomputing embeddings from the training set "
                        "to keep KATE retrieval aligned."
                    )
                    complaint_embeddings = self.symptom_encoder.encode(self.training_set["Chief_complain"].fillna("").tolist())
                    diagnosis_embeddings = self.symptom_encoder.encode(self.training_set["Diagnosis in ED"].fillna("").tolist())
                self.complaint_embeddings = complaint_embeddings
                self.diagnosis_embeddings = diagnosis_embeddings
                self.embeddings_cache = np.concatenate([complaint_embeddings, diagnosis_embeddings], axis=1)
                self.ktas_retriever = KTASSingleCaseRetriever(
                    training_set=self.training_set,
                    complaint_embeddings=complaint_embeddings,
                    diagnosis_embeddings=diagnosis_embeddings,
                    symptom_encoder=self.symptom_encoder,
                )
            else:
                if k_shots_ablation:
                    print("Using full training set embeddings")
                    embeddings_path = utils._DATASETS[dataset]['full_training_embeddings_filepath']
                else:
                    embeddings_path = utils._DATASETS[dataset]['training_embeddings_filepath']
                self.embeddings_cache = np.load(embeddings_path,allow_pickle=True)
    
    def predict(self, *, row, model, k_shots, return_json, serialization_strategy, vitals_off, bias, debug=False, worst_example=False):
        """
        Route prediction based on strategy.
        """
        if self.strategy in ["FewShot", "FewShotCoT"]:
            return self.few_shot_prediction(row=row, model=model, k_shots=k_shots, return_json=return_json, serialization=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug, worst_example=worst_example)
        elif self.strategy in ["KATE", "KATECoT", "KATEAutoCoT"]:
            return self.kate_prediction(row=row, model=model, k_shots=k_shots, return_json=return_json, serialization=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug, worst_example=worst_example)
        elif self.strategy in ["KATE2Call"]:
            return self.kate_two_call_prediction(row=row, model=model, k_shots=k_shots, serialization=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug, worst_example=worst_example)
        elif self.strategy in ["Vanilla", "Vanillav0", "AutoCoT", "CoT","DemonstrationCoT"]:
            return self.zero_shot_prediction(row=row, model=model, return_json=return_json, serialization=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug)
        elif self.strategy in ["SelfConsistency"]:
            return self.self_consistency_prediction(row=row, model=model, return_json=return_json, serialization=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug)
        elif self.strategy in ["MultiAgent"]:
            return self.multi_agent_prediction(row=row, model=model, return_json=return_json, serialization=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def zero_shot_prediction(self, *, row, model, return_json=False, serialization='natural', vitals_off=False, bias=False, debug=False, **kwargs):
        """Zero-shot prediction implementation."""

        prompt = prompts.format_instruction_prompt_for_blackbox(
            row=row,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=return_json,
            serialization=serialization
        )
        
        if self.hippa:
            response = utils.query_gpt_safe(prompt, model=model, return_json=return_json, debug=debug, is_prompt_full=True)
        else:
            response = utils.query_llm_full(prompt, model=model, return_json=return_json, debug=debug)
        
        return prompt, response, {}

    def few_shot_prediction(self, *, row, model, k_shots=5, return_json=False, serialization='natural', vitals_off=False, bias=False, debug=False, worst_example=False, **kwargs):
        """Few-shot prediction implementation."""
        if self.training_set is None:
            raise ValueError("Few-shot strategy requires a training set of examples.")

        _, examples = utils.stratified_df(self.training_set, target_col=self.target, test_size=k_shots)

        if worst_example:
            examples = examples.iloc[[-1]]

        formatted_examples = "\n\n".join([
            self._format_example_with_reasoning(example, serialization)
            for _, example in examples.iterrows()
        ])
        
        prompt = prompts.format_instruction_prompt_for_blackbox(
            row=row,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=return_json,
            serialization=serialization,
            examples=formatted_examples
        )
        
        if self.hippa:
            response = utils.query_gpt_safe(prompt, model=model, return_json=return_json, debug=debug, is_prompt_full=True)
        else:
            response = utils.query_llm_full(prompt, model=model, return_json=return_json, debug=debug)
        
        return prompt, response, {"Retrieved_Examples": formatted_examples}

    def self_consistency_prediction(self, *, row, model, return_json=False, serialization='natural', vitals_off=False, bias=False, debug=False, num_trials=5, **kwargs):
        """Self-consistency prediction implementation."""
        responses = []
        
        for _ in range(num_trials):
            prompt = prompts.format_instruction_prompt_for_blackbox(
                row=row,
                strategy='SelfConsistency',
                dataset=self.dataset.lower(),
                return_json=True,
                serialization=serialization
            )
            
            if self.hippa:
                response = utils.query_gpt_safe(prompt, temperature = 1.0, model=model, return_json=True, debug=debug, is_prompt_full=True)
            else:
                response = utils.query_llm_full(prompt, temperature = 1.0, model=model, return_json=True, debug=debug)
            
            try:
                response_data = json.loads(response)
                responses.append(response_data['Acuity'])
            except (json.JSONDecodeError, KeyError) as e:
                if debug:
                    print(f"Error parsing response: {e}")
                    print(f"Raw response: {response}")
                continue
        
        if not responses:
            raise ValueError("No valid responses received from any trial")
        
        from statistics import mode
        final_acuity = mode(responses)
        
        if return_json:
            final_response = json.dumps({
                "Acuity": final_acuity,
                "Reasoning": f"Based on {len(responses)} trials, the most common prediction was {final_acuity}. All predictions: {responses}"
            })
        else:
            final_response = f"Based on {len(responses)} trials, the most common prediction was {final_acuity}. All predictions: {responses}"
        
        return prompt, final_response, {}

    def _format_example_with_reasoning(self, example, serialization):
        """Format a single example with optional reasoning."""
        text = prompts.format_row(example, dataset=self.dataset.lower(), serialization=serialization)
        text += f"\nAcuity Level: {example[self.target]}"
        reasoning = example.get('Reasoning', '')
        if pd.notna(reasoning) and str(reasoning).strip():
            text += f"\nReasoning: {reasoning}"
        return text

    def kate_prediction(self, *, row, model, k_shots=5, return_json=False, serialization='natural', vitals_off=False, bias=False, debug=False, worst_example=False, **kwargs):
        """KATE prediction implementation."""
        if self.training_set is None:
            raise ValueError("KATE strategy requires a training set of examples.")

        examples = self._retrieve_top_k_examples(row, k_shots, worst_embedding=worst_example)

        if worst_example:
            examples = examples.iloc[[-1]]

        formatted_examples = "\n\n".join([
            self._format_example_with_reasoning(example, serialization)
            for _, example in examples.iterrows()
        ])
        
        prompt = prompts.format_instruction_prompt_for_blackbox(
            row=row,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=return_json,
            serialization=serialization,
            examples=formatted_examples
        )
        
        if self.hippa:
            response = utils.query_gpt_safe(prompt, model=model, return_json=return_json, debug=debug, is_prompt_full=True)
        else:
            response = utils.query_llm_full(prompt, model=model, return_json=return_json, debug=debug)
        
        return prompt, response, {"Retrieved_Examples": formatted_examples}
    
    def kate_two_call_prediction(self, *, row, model, k_shots=5, serialization='natural', vitals_off=False, bias=False, debug=False, worst_example=False, **kwargs):
        """KATE two-call prediction: free-text reasoning first, then extract acuity with a second call."""
        if self.training_set is None:
            raise ValueError("KATE2Call strategy requires a training set of examples.")

        examples = self._retrieve_top_k_examples(row, k_shots, worst_embedding=worst_example)

        if worst_example:
            examples = examples.iloc[[-1]]

        formatted_examples = "\n\n".join([
            self._format_example_with_reasoning(example, serialization)
            for _, example in examples.iterrows()
        ])

        # Call 1: free-text reasoning (no JSON constraint)
        prompt = prompts.format_instruction_prompt_for_blackbox(
            row=row,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=False,
            serialization=serialization,
            examples=formatted_examples
        )

        reasoning = utils.query_llm_full(prompt, model=model, return_json=False, debug=debug)

        # Call 2: extract acuity from reasoning using same model
        extract_prompt = {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": (
                f"Based on the following reasoning, output the final KTAS acuity level as a single integer from 1 to 5. "
                f"Output the number only, nothing else.\n\nReasoning:\n{reasoning}"
            )
        }

        acuity_response = utils.query_llm_full(extract_prompt, model=model, return_json=False, debug=debug)

        # Return reasoning as the response, acuity_response in metadata for parsing
        combined_response = f"{reasoning}\n\nACUITY: {acuity_response}"

        return prompt, combined_response, {
            "Retrieved_Examples": formatted_examples,
            "Acuity_Response": acuity_response,
        }

    def get_top_k_similar(self, embedding, embeddings, k):
        """
        Find the top-k most similar samples to a given embedding.
        """
        similarities = np.dot(embeddings, embedding)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices, similarities[top_k_indices]

    def _retrieve_top_k_examples(self, row, k, worst_embedding=False):
        """
        Retrieve the top K most similar examples to the prompt based on Ada embeddings.
        First searches by concatenated chief complaint and diagnosis embeddings, then refines by vital signs.

        :param row: The input row containing chief complaint and vital signs.
        :param k: Number of examples to retrieve.
        :param worst_embedding: If True, retrieve worst examples instead of best.
        :return: DataFrame of top K examples.
        """
        # Get embeddings based on dataset type
        if self.dataset.lower() == 'triage-mimic':
            symptom = row['chiefcomplaint']
            combined_embedding = self.symptom_encoder.encode(symptom)
            # Search using combined embedding
            top_k_indices, _ = self.get_top_k_similar(combined_embedding, self.embeddings_cache, k*3, reverse=worst_embedding)

        elif self.dataset.lower() == 'triage-ktas':
            return self.ktas_retriever.retrieve_top_k_examples(row, k, worst_embedding=worst_embedding)
            
        elif self.dataset.lower() == 'triage-handbook':
            symptom = row['Clinical Vignettes']
            embeddings = self.symptom_encoder.encode(symptom)
            top_k_indices, _ = self.get_top_k_similar(embeddings, self.embeddings_cache, k*3)
            top_k_samples = self.training_set.loc[top_k_indices]
            # print(top_k_samples)
            return top_k_samples
        
        # Then, narrow down to k by vitals
        if self.dataset.lower() == 'triage-mimic':
            vital_sign_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
        elif self.dataset.lower() == 'triage-ktas':
            vital_sign_columns = ['NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT']
        
        numeric_vital_signs = [col for col in vital_sign_columns if pd.api.types.is_numeric_dtype(row[col])]

        query_vital_signs = row[numeric_vital_signs].values.astype(float)
        top_25_vital_signs = self.training_set.loc[top_k_indices, numeric_vital_signs]
        top_25_vital_signs = top_25_vital_signs.apply(pd.to_numeric, errors='coerce')  # Coerce invalid values to NaN
        top_25_vital_signs = top_25_vital_signs.dropna()  # Drop rows with NaN values
        
        # Ensure the resulting values are numpy floats
        top_25_vital_signs = top_25_vital_signs.values.astype(float)
        query_vital_signs_normalized = query_vital_signs / np.linalg.norm(query_vital_signs)
        top_25_vital_signs_normalized = top_25_vital_signs / np.linalg.norm(top_25_vital_signs, axis=1, keepdims=True)
        
        vital_signs_similarity = np.dot(top_25_vital_signs_normalized, query_vital_signs_normalized)
        top_k_vital_indices = np.argsort(vital_signs_similarity)[-k:][::-1]
        
        # Retrieve top-k samples
        top_k_vital_samples = self.training_set.loc[top_k_indices[top_k_vital_indices]]
        
        # # Add the 'pain' column back to the result (unchanged)
        # top_k_vital_samples['pain'] = self.training_set.loc[top_k_indices[top_k_vital_indices], 'pain']
        
        return top_k_vital_samples
    
