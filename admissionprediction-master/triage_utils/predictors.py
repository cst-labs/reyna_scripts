import numpy as np
from triage_utils import core as utils
from triage_utils import prompts
from sentence_transformers import SentenceTransformer
import pandas as pd
import json


def _clean_retrieval_text(chief_complaint, past_medical_history):
    parts = []
    for label, value in (
        ("Chief complaint", chief_complaint),
        ("Past medical history", past_medical_history),
    ):
        if pd.isna(value):
            continue
        text = str(value).replace("|", ", ").strip()
        if text:
            parts.append(f"{label}: {text}.")
    return " ".join(parts)

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
        if training_set is not None and "KATE" in self.strategy:
            model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
            self.symptom_encoder = SentenceTransformer(model_name, local_files_only=True)
            if self.dataset.lower() == 'triage-esi-cleaned':
                embeddings_path = utils._DATASETS[dataset]['training_embeddings_filepath']
                self.embeddings_cache = np.load(embeddings_path, allow_pickle=True)
            elif self.dataset.lower() == 'triage-ktas':
                complaint_embeddings = np.load(utils._DATASETS[dataset]['training_complaint_embeddings_filepath'], allow_pickle=True)
                diagnosis_embeddings = np.load(utils._DATASETS[dataset]['training_diagnosis_embeddings_filepath'], allow_pickle=True)
                self.complaint_embeddings = complaint_embeddings
                self.diagnosis_embeddings = diagnosis_embeddings
                self.embeddings_cache = np.concatenate([complaint_embeddings, diagnosis_embeddings], axis=1)
            else:
                if k_shots_ablation:
                    print("Using full training set embeddings")
                    embeddings_path = utils._DATASETS[dataset]['full_training_embeddings_filepath']
                else:
                    embeddings_path = utils._DATASETS[dataset]['training_embeddings_filepath']
                self.embeddings_cache = np.load(embeddings_path,allow_pickle=True)
    
    def predict(self, *, row, model, k_shots, return_json, serialization_strategy, vitals_off, bias, debug=False):
        """
        Route prediction based on strategy.
        """
        if self.strategy in ["FewShot", "FewShotCoT"]:
            return self.few_shot_prediction(row=row, model=model, k_shots=k_shots, return_json=return_json, serialization_strategy=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug)
        elif self.strategy in ["KATE", "KATECoT", "KATEAutoCoT"]:
            return self.kate_prediction(row=row, model=model, k_shots=k_shots, return_json=return_json, serialization_strategy=serialization_strategy, vitals_off=vitals_off, bias=bias, debug=debug)
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
        
        return prompt, response

    def few_shot_prediction(self, *, row, model, k_shots=5, return_json=False, serialization='natural', vitals_off=False, bias=False, debug=False, **kwargs):
        """Few-shot prediction implementation."""
        if self.training_set is None:
            raise ValueError("Few-shot strategy requires a training set of examples.")
        
        _, examples = utils.stratified_df(self.training_set, target_col=self.target, test_size=k_shots)
        
        formatted_examples = "\n\n".join([
            prompts.format_row(example, dataset=self.dataset.lower(), serialization=serialization) + 
            f"\nAcuity Level: {example[self.target]}"
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
        
        return prompt, response

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
        
        return prompt, final_response

    def kate_prediction(self, *, row, model, k_shots=5, return_json=False, serialization='natural', vitals_off=False, bias=False, debug=False, **kwargs):
        """KATE prediction implementation."""
        if self.training_set is None:
            raise ValueError("KATE strategy requires a training set of examples.")
        
        examples = self._retrieve_top_k_examples(row, k_shots)
        
        formatted_examples = "\n\n".join([
            prompts.format_row(example, dataset=self.dataset.lower(), serialization=serialization) + 
            f"\nAcuity Level: {example[self.target]}"
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
        
        return prompt, response
    
    def get_top_k_similar(self, embedding, embeddings, k):
        """
        Find the top-k most similar samples to a given embedding.
        """
        similarities = np.dot(embeddings, embedding)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices, similarities[top_k_indices]

    def _rerank_by_vitals(self, row, candidate_indices, vital_sign_columns, k):
        query_vital_signs = pd.to_numeric(row[vital_sign_columns], errors='coerce').values.astype(float)
        candidate_vitals = self.training_set.loc[candidate_indices, vital_sign_columns]
        candidate_vitals = candidate_vitals.apply(pd.to_numeric, errors='coerce').dropna()
        if candidate_vitals.empty:
            return self.training_set.loc[candidate_indices[:k]]

        candidate_indices = candidate_vitals.index.to_numpy()
        candidate_vitals = candidate_vitals.values.astype(float)

        query_norm = np.linalg.norm(query_vital_signs)
        candidate_norms = np.linalg.norm(candidate_vitals, axis=1)
        valid_mask = (candidate_norms > 0) & np.isfinite(candidate_norms)
        if query_norm == 0 or not np.isfinite(query_norm) or not valid_mask.any():
            return self.training_set.loc[candidate_indices[:k]]

        query_normalized = query_vital_signs / query_norm
        candidate_normalized = candidate_vitals[valid_mask] / candidate_norms[valid_mask][:, None]
        candidate_indices = candidate_indices[valid_mask]
        similarities = np.dot(candidate_normalized, query_normalized)
        top_indices = np.argsort(similarities)[-k:][::-1]
        return self.training_set.loc[candidate_indices[top_indices]]

    def _retrieve_top_k_examples(self, row, k):
        """
        Retrieve the top K most similar examples to the prompt based on Ada embeddings.
        First searches by concatenated chief complaint and diagnosis embeddings, then refines by vital signs.

        :param row: The input row containing chief complaint and vital signs.
        :param k: Number of examples to retrieve.
        :return: DataFrame of top K examples.
        """
        # Get embeddings based on dataset type
        if self.dataset.lower() == 'triage-esi-cleaned':
            symptom = _clean_retrieval_text(row.get('chief_complaint'), row.get('past_medical_history'))
            combined_embedding = self.symptom_encoder.encode(symptom)
            top_k_indices, _ = self.get_top_k_similar(combined_embedding, self.embeddings_cache, k * 3)
        elif self.dataset.lower() == 'triage-mimic':
            symptom = row['chiefcomplaint']
            combined_embedding = self.symptom_encoder.encode(symptom)
            # Search using combined embedding
            top_k_indices, _ = self.get_top_k_similar(combined_embedding, self.embeddings_cache, k*3)
            
        elif self.dataset.lower() == 'triage-ktas':
            # Handle cases where either symptom or diagnosis is empty/null
            symptom = row['Chief_complain']
            diagnosis = row['Diagnosis in ED']
            
            if pd.isna(symptom) or symptom == '':
                # Only use diagnosis embedding if symptom is empty
                combined_embedding = self.symptom_encoder.encode(diagnosis)
                top_k_indices, _ = self.get_top_k_similar(combined_embedding, self.diagnosis_embeddings, k*3)
            elif pd.isna(diagnosis) or diagnosis == '':
                # Only use symptom embedding if diagnosis is empty  
                combined_embedding = self.symptom_encoder.encode(symptom)
                top_k_indices, _ = self.get_top_k_similar(combined_embedding, self.complaint_embeddings, k*3)
            else:
                # Combine both embeddings if neither is empty
                symptom_embedding = self.symptom_encoder.encode(symptom)
                diagnosis_embedding = self.symptom_encoder.encode(diagnosis)
                combined_embedding = np.concatenate([symptom_embedding, diagnosis_embedding])
                # Search using embedding
                top_k_indices, _ = self.get_top_k_similar(combined_embedding, self.embeddings_cache, k*3)
            
        elif self.dataset.lower() == 'triage-handbook':
            symptom = row['Clinical Vignettes']
            embeddings = self.symptom_encoder.encode(symptom)
            top_k_indices, _ = self.get_top_k_similar(embeddings, self.embeddings_cache, k*3)
            top_k_samples = self.training_set.loc[top_k_indices]
            # print(top_k_samples)
            return top_k_samples
        
        # Then, narrow down to k by vitals
        if self.dataset.lower() == 'triage-esi-cleaned':
            vital_sign_columns = [
                'triage_vital_temp',
                'triage_vital_hr',
                'triage_vital_rr',
                'triage_vital_o2',
                'triage_vital_sbp',
                'triage_vital_dbp',
            ]
            return self._rerank_by_vitals(row, top_k_indices, vital_sign_columns, k)
        if self.dataset.lower() == 'triage-mimic':
            vital_sign_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
        elif self.dataset.lower() == 'triage-ktas':
            vital_sign_columns = ['NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT']
        return self._rerank_by_vitals(row, top_k_indices, vital_sign_columns, k)
    
