from typing import Union, List, Dict

import openai
import litellm
from transformers import GenerationConfig

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

	except openai.OpenAIError as e:
		# Error handling
		should_retry = litellm._should_retry(e.status_code)
		print("Error: API failed to respond.", e, f"should_retry: {should_retry}")
		new_texts = []

	return new_texts