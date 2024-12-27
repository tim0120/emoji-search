import os
from typing import Union, List, Dict

import litellm
import numpy as np
import torch
from torch.nn import functional as F
from transformers import GenerationConfig, AutoTokenizer, AutoModel

def local_embed(input_texts: List[str], tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
	with torch.no_grad():
		batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
		outputs = model(**batch_dict)
	embeddings = outputs.last_hidden_state[:, 0]
	embeddings = F.normalize(embeddings, p=2, dim=1)
	embeddings = embeddings.cpu().numpy()
	return embeddings

def api_generate(
	prompts: Union[List[str], List[List[Dict[str, str]]]],
	model: str,
	generation_config: GenerationConfig = GenerationConfig(
		max_new_tokens=2048
	),
	num_retries: int = 4,
) -> List[str]:
	"""
	This is a helper function to make it easy to generate using various LLM APIs
	(e.g. OpenAI, Anthropic, etc.) with built in error-handling.
	"""
	litellm.suppress_debug_info = True
	# If we pass a list of prompts, convert to message format
	if isinstance(prompts[0], str):
		prompts = [[{"role": "user", "content": p}] for p in prompts]
	try:
		# Attempt batched completion call with litellm
		responses = litellm.batch_completion(
			model=model,
			messages=prompts,
			top_p=generation_config.top_p,
			max_tokens=generation_config.max_new_tokens,
			num_retries=num_retries,
		)
		new_texts = [r.choices[0].message.content for r in responses]

	except Exception as e:
		# Error handling
		should_retry = litellm._should_retry(e.status_code)
		print("Error: API failed to respond.", e, f"should_retry: {should_retry}")
		new_texts = []

	return new_texts

def api_embed(input: Union[str, List[str]], model: str, is_hf_model: bool = False) -> np.ndarray:
	"""
	This is a helper function to get text embeddings using various LLM APIs
	with built in error-handling.
	"""
	litellm.suppress_debug_info = True
	input = [input] if isinstance(input, str) else input
	try:
		if 'openai/' in model:
			model = model.split('/')[1]
		if is_hf_model:
			model = f"huggingface/{model}"
		data = litellm.embedding(model=model, input=input, num_retries=10).data
		embeddings = [datum['embedding'] for datum in data]

	except Exception as e:
		# Error handling
		print("Error: API failed to respond.", e)
		embeddings = []
	
	return np.array(embeddings)