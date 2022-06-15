#!/usr/bin/env python
import gc
from multiprocessing import context
import time
import json
from typing import Any, Callable, List, Tuple, Union
import torch
from megatron.sampling import nucleus_filtering, repetition_filtering, tailfree_filtering, temperature_filtering, top_a_filtering, top_k_filtering, typical_filtering
from megatron.utils import print_rank_0, setup_for_inference_or_eval, is_mp_rank_0
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.exceptions import HTTPException
import uvicorn
import threading
import functools
import asyncio
from megatron import mpu
from megatron.initialize import _set_random_seed

from megatron.text_generation_utils import (
	stream_tokens,
	stream_tokens2,
)


def main():
	model, neox_args = setup_for_inference_or_eval(use_cache=True, overwrite_values={'seed': int(time.time())})

	# This is needed.  For some reason the seed doesn't get set properly, even though megatron's code is supposed to set it.
	torch.manual_seed(int(time.time()))

	if neox_args.recompute:
		model.module.inference_mode(
			use_cache=False
		)  # don't use kv cache if recomputing
	
	# For future tests, with seed set to 1235:
	# input: [13841]*1024
	# n_samples=1, maximum_tokens=32, eos_token_id=neox_args.tokenizer.eod, temperature=0.7, top_k=40, top_p=0.0, n_gens_per_context=16
	# Expected tokens: tensor([[  310,    13,   417,   281,   320,   247,  1077,  1345,  3908,   273,1329,   323,   253,  3128,   275,   436,  8881,    15,   831,   310,271,  7880,   285,   440, 28758,    13,   533,   253, 36309,   403,1029,   449,   187,   187,   510]])
	# Expected probabilities: tensor([[0.0118, 0.0065, 0.0043, 0.0493, 0.4554, 0.1039, 0.0342, 0.0124, 0.0271,0.4225, 0.1006, 0.8736, 0.7500, 0.0258, 0.1132, 0.0398, 0.0704, 0.6807,0.0060, 0.7894, 0.0493, 0.0029, 0.0378, 0.3682, 0.1915, 0.0184, 0.0117,0.0723, 0.1320, 0.7931, 0.2952, 0.0067, 0.9486, 0.9989, 0.2800]])
	#
	# We can maintain consistency between tests by calling (before each test):
	# _set_random_seed(neox_args.seed)
	# torch.manual_seed(neox_args.seed)

	# Start up API server
	if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
		print("Starting API server...")
		app = Starlette(debug=True, routes=[
			Route('/', homepage, methods=['POST']),
		])
		app.state.model_lock = threading.Lock()
		app.state.model = model
		app.state.neox_args = neox_args

		uvicorn.run(app, host="0.0.0.0", port=5034, log_level="info", use_colors=False)
	else:
		while True:
			prompt_runner_2(neox_args, model, "EMPTY TEXT", [], 0, 0, None, 0)
	
	return


async def homepage(request):
	print("Got a request!")
	params = await request.json()

	filters = []
	for filter in params['filters']:
		if 'top_k' in filter:
			filters.append(functools.partial(top_k_filtering, param=int(filter['top_k'])))
		elif 'top_p' in filter:
			filters.append(functools.partial(nucleus_filtering, param=float(filter['top_p'])))
		elif 'repetition' in filter:
			filters.append(functools.partial(repetition_filtering, param=float(filter['repetition']['param']), max_context=int(filter['repetition']['range'])))
		elif 'typical' in filter:
			filters.append(functools.partial(typical_filtering, param=float(filter['typical'])))
		elif 'temp' in filter:
			filters.append(functools.partial(temperature_filtering, param=float(filter['temp'])))
		elif 'tailfree' in filter:
			filters.append(functools.partial(tailfree_filtering, param=float(filter['tailfree'])))
		elif 'top_a' in filter:
			filters.append(functools.partial(top_a_filtering, param=float(filter['top_a'])))
		else:
			raise HTTPException("Unknown filter: " + filter['name'])

	loop = asyncio.get_event_loop()
	func = functools.partial(prompt_runner,
		lock=request.app.state.model_lock,
		model=request.app.state.model,
		neox_args=request.app.state.neox_args,
		prompt=params['prompt'],
		filters=filters,
		maximum_tokens=int(params['maximum_tokens']),
		n_samples=int(params['n_samples']),
		eos_token_id=int(params['eos_token_id']) if params['eos_token_id'] else None,
		n_gens_per_context=int(params['n_gens_per_context']),
	)
	generated_tokens, generated_probs = await loop.run_in_executor(None, func)
	return JSONResponse({
		'tokens': generated_tokens,
		'probabilities': generated_probs,
	})


def prompt_runner(lock, model, neox_args, prompt, filters, maximum_tokens, n_samples, eos_token_id, n_gens_per_context):
	print("Grabbing lock...")
	with lock:
		print("Generating sample...")
		result = prompt_runner_2(
			neox_args=neox_args,
			model=model,
			prompt=prompt,
			filters=filters,
			maximum_tokens=maximum_tokens,
			n_samples=n_samples,
			eos_token_id=eos_token_id,
			n_gens_per_context=n_gens_per_context,
		)
		print("Sample generated!")
		return result


def prompt_runner_2(
	neox_args,
	model,
	prompt: str,
	filters: List[Callable],
	maximum_tokens: int,
	n_samples: int,
	eos_token_id: Union[int, None],
	n_gens_per_context: int,
):
	if eos_token_id is None:
		eos_token_id = neox_args.tokenizer.eod
	
	#print(f"Device {torch.cuda.current_device()} waiting for broadcast config with maximum_tokens = {maximum_tokens}...")
	#print(f"Device {torch.cuda.current_device()} Before: {maximum_tokens}, {n_samples}, {eos_token_id}, {n_gens_per_context}")
	maximum_tokens, n_samples, eos_token_id, n_gens_per_context, filters = broadcast_objects([maximum_tokens, n_samples, eos_token_id, n_gens_per_context, filters])
	#print(f"Device {torch.cuda.current_device()} After: {maximum_tokens}, {n_samples}, {eos_token_id}, {n_gens_per_context}")
	#print(f"Device {torch.cuda.current_device()} broadcasted config with maximum_tokens = {maximum_tokens}!")

	n_gens_per_context = min(n_gens_per_context, neox_args.seq_length)

	if len(prompt) == 0:
		context_tokens = [neox_args.tokenizer.eod]
	else:
		context_tokens = neox_args.tokenizer.tokenize(prompt)
	
	# Truncate context if it's too long
	if len(context_tokens) > neox_args.seq_length:
		context_tokens = context_tokens[-neox_args.seq_length:]
	
	for (
		generated_tokens,
		generated_token_probabilities,
		generation_end_index,
		_,
	) in stream_tokens2(
		neox_args=neox_args,
		model=model,
		context_tokens=context_tokens[:neox_args.seq_length],
		eos_token_id=eos_token_id,
		maximum_tokens=maximum_tokens,
		recompute=neox_args.recompute,
		filters=filters,
		n_samples=n_samples,
		n_gens_per_context=n_gens_per_context,
	):
		pass

	if mpu.get_model_parallel_rank() == 0:
		generated_text = []
		generated_probabilities = []
		for i in range(n_samples):
			generated_text.append(neox_args.tokenizer.detokenize(generated_tokens[i][:generation_end_index[i] + 1].cpu().tolist()))
			generated_probabilities.append(generated_token_probabilities[i][:generation_end_index[i] + 1].cpu().tolist())
		return generated_text, generated_probabilities
	else:
		return None, None


def broadcast_objects(objs: List[Any]) -> List[Any]:
	torch.distributed.broadcast_object_list(objs, src=mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
	return objs


if __name__ == "__main__":
	main()
