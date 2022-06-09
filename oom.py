#!/usr/bin/env python
import torch
from megatron.utils import setup_for_inference_or_eval
from megatron import mpu

from megatron.text_generation_utils import generate_samples_from_prompt


def main():
	model, neox_args = setup_for_inference_or_eval(use_cache=True)

	if neox_args.recompute:
		model.module.inference_mode(
			use_cache=False
		)  # don't use kv cache if recomputing
	
	# Start up API server
	if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
		for i in range(16):
			prompt = neox_args.tokenizer.detokenize([10482]*(2030 + i))
			torch.distributed.barrier(group=mpu.get_model_parallel_group())
			report_memory(f"Before iter {i}")
			result = generate_samples_from_prompt(neox_args, model, prompt, recompute=neox_args.recompute, temperature=0.7, top_k=40)
			print(f"Generation Time: {result[0]['duration_seconds']}")
			report_memory(f"After iter {i}")
	else:
		for i in range(16):
			torch.distributed.barrier(group=mpu.get_model_parallel_group())
			report_memory(f"Before iter {i}")
			generate_samples_from_prompt(neox_args, model, "EMPTY TEXT", recompute=neox_args.recompute, temperature=0.7, top_k=40)
			report_memory(f"After iter {i}")


def report_memory(name):
	"""Simple GPU memory report."""
	mega_bytes = 1024.0 * 1024.0
	string = name + " memory (MB)"
	string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
	string += " | max allocated: {}".format(
		torch.cuda.max_memory_allocated() / mega_bytes
	)
	string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
	string += " | max reserved: {}".format(
		torch.cuda.max_memory_reserved() / mega_bytes
	)
	print(f"Memory Report, Device {torch.cuda.current_device()}: {string}")


if __name__ == "__main__":
	main()
