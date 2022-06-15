from typing import Callable, List
import torch
import torch.nn.functional as F


def sample_tokens(logits, context, filters: List[Callable]):
	"""
	Given logits of shape [batch_size, vocab_size], returns tuple (indices, probabilities), each of shape [batch_size]
	Probability being the liklihood of the selected token.  Useful for debugging the filters and their settings.
	i.e. If generation goes awry, the user can inspect the probs and see if a lot of them are low, and thus know to tweak the filters.
	"""
	# Convert to float for better accuracy in the filter calculations
	logits = logits.float()

	for filter in filters:
		logits = filter(logits, context)
	
	probs = F.softmax(logits, dim=-1)
	generated_tokens = torch.multinomial(probs, num_samples=1).view(-1)

	return (generated_tokens, torch.gather(probs, 1, generated_tokens[:,None]).view(-1))


def top_a_filtering(logits, context, param: float, filter_value: float = -float("Inf")):
	"""
	Filter tokens dynamically based on the probability of the most likely token.

	We expect param to be between 0.0 and 1.0 (off).

	Top-A dynamically calculates a probability threshold, based on the probability of the top token,
	using the formula threshold = max(probabilities)**2 * param.
	Any tokens below this threshold are filtered out.
	If the top token has a very high probability, Top-A will tend to filter out the vast majority of other tokens.
	This cooresponds to cases like factual contexts, where the model is (hopefully) very confident in correct answer token.
	But in cases where the top token has only a moderate probability, Top-A will tend to include a lot more tokens.
	"""
	if param == 1.0:
		return logits

	logits = torch.clone(logits)

	probs = F.softmax(logits, dim=-1)
	limit = (torch.max(probs, dim=-1, keepdim=True).values ** 2) * (1.0 - param)
	indicies_to_remove = probs < limit
	logits[indicies_to_remove] = filter_value

	return logits


def typical_filtering(logits, context, param: float, filter_value: float = -float("Inf")):
	"""
	See: https://arxiv.org/abs/2202.00666

	param should be between 0.0 and 1.0 (off)
	"""
	if param == 1.0:
		return logits

	normalized = torch.nn.functional.log_softmax(logits, dim=-1)
	probs = torch.exp(normalized)
	ent = -(normalized * probs).nansum(-1, keepdim=True)

	# shift and sort
	shifted_scores = torch.abs((-normalized) - ent)
	_, sorted_indices = torch.sort(shifted_scores, descending=False)
	cumulative_probs = probs.gather(-1, sorted_indices).cumsum(dim=-1)

	# Remove tokens with cumulative mass above the threshold
	sorted_indices_to_remove = cumulative_probs > param
	# Shift the indices to the right to keep also the first token above the threshold
	sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
	sorted_indices_to_remove[..., 0] = 0
	indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

	return logits.masked_fill(indices_to_remove, filter_value)


def tailfree_filtering(logits, context, param: float, filter_value: float = -float("Inf")):
	"""
	Don't know if I implemented this correctly.
	The author's code is here: https://github.com/TrentBrick/TailFreeSampling/blob/29d2049fa9326879474a03e9c81cdd639d01dc87/sampling.py
	But, just reading the code, it doesn't seem correct at the time of this writing.
	tail_ids has indexes in the "sorted" space, but then uses it to calculate logit_inds.
	Thus logit_inds would be in sorted space.  Yet it never translates from sorted space back
	to logit space.
	So I tried implementing it my own way below.
	"""
	if param == 1.0:
		return logits

	probs = F.softmax(logits, dim=-1)
	sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

	grad = sorted_probs[:,1:] - sorted_probs[:,:-1] # first derivative
	grad = grad[:,1:] - grad[:,:-1] # 2nd derivative
	abs_grad = torch.abs(grad)
	norm_grad = abs_grad / torch.sum(abs_grad, dim=-1, keepdim=True)  # Normalize
	cumulative_grad = norm_grad.cumsum(dim=-1)

	sorted_indices_to_remove = cumulative_grad > param
	# Shift the indicies, because gradients aren't aligned to original probabilities
	# NOTE: Not sure if the shift should be 1 or 2?
	sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (0, 2), value=True)
	sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
	sorted_indices_to_remove[..., 0] = 0
	indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

	return logits.masked_fill(indices_to_remove, -float("Inf"))


def nucleus_filtering(logits, context, param: float, filter_value: float = -float("Inf")):
	"""
	param should be between 0.0 and 1.0 (off)
	"""
	if param == 1.0:
		return logits

	# convert to 1D
	sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
	cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

	# Remove tokens with cumulative probability above the threshold
	sorted_indices_to_remove = cumulative_probs > param
	# Shift the indices to the right to keep also the first token
	# above the threshold
	sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
	sorted_indices_to_remove[..., 0] = 0
	indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
	return logits.masked_fill(indices_to_remove, filter_value)


def top_k_filtering(logits, context, param: int, filter_value: float = -float("Inf")):
	if param == 0:
		return logits
	
	indices_to_remove = logits < torch.topk(logits, param)[0][..., -1, None]
	return logits.masked_fill(indices_to_remove, filter_value)


def temperature_filtering(logits, context, param: float):
	if param == 1.0:
		return logits

	return logits / param


def repetition_filtering(logits, context, param: float, max_context: int):
	"""
	This implementation follows HuggingFace's lead as of now (2022.06.15).
	However, it should be noted that both HF's implementation and the original paper that introduced repetition penalty
	after a "problem".  They apply the penalty to the raw logits, which are not normalized.  i.e. they have no scale.
	That's why HF had to add the "<0" condition to their implementation, since the paper's original formula would actually
	_boost_ negative logits when it clearly meant to penalize them.
	But HF's implementation, for example, fails for a logit of 0, applying no penalty at all.
	The penalty should clearly be applied to the probability space, not logits.  But what formula would be best, I have not
	researched yet.
	"""
	if param == 1.0:
		return logits

	context = context[:, -max_context:]
	x = torch.gather(logits, 1, context)
	x = torch.where(x < 0, x * param, x / param)

	return logits.scatter(1, context, x)