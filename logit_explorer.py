#!/usr/bin/env python
import gc
from multiprocessing import context
import time
import json
from typing import Any, List, Tuple, Union
import torch
from megatron.utils import print_rank_0, setup_for_inference_or_eval, is_mp_rank_0
import threading
import functools
import asyncio
from megatron import mpu
from megatron.initialize import _set_random_seed
import numpy as np

from megatron.text_generation_utils import (
	stream_tokens,
	stream_tokens2,
)


def main():
	model, neox_args = setup_for_inference_or_eval(use_cache=True)

	# This is needed.  For some reason the seed doesn't get set properly, even though megatron's code is supposed to set it.
	torch.manual_seed(neox_args.seed)

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


	eos_token_id = neox_args.tokenizer.eod
	temperature = 0.7
	top_k = 40
	top_p = 0.0

	prompt = neox_args.tokenizer.tokenize(BAD_FACT_PROMPT)

	logits = []
	
	for (
		generated_tokens,
		generated_token_probabilities,
		generated_logits,
		generation_end_index,
		_,
	) in stream_tokens2(
		neox_args=neox_args,
		model=model,
		context_tokens=prompt,
		eos_token_id=eos_token_id,
		maximum_tokens=128,
		recompute=neox_args.recompute,
		temperature=temperature,
		top_k=top_k,
		top_p=top_p,
		n_samples=1,
		n_gens_per_context=128,
		return_logits=True,
	):
		if is_mp_rank_0():
			logits.append(generated_logits[0,-1,:].cpu().numpy())
	
	if is_mp_rank_0():
		logits = np.stack(logits, axis=0)
		print(f"logits: {logits.shape}")
		print(f"generated_tokens: {generated_tokens.shape}")
		with open('logits3.npz', 'wb') as f:
			np.savez(f, logits=logits, generated_tokens=generated_tokens[0].cpu().numpy())



BOOK_TITLE_PROMPT = """
[Book Titles]
Some Women's Ways [Short stories.]
Adventure Square. Poems
Playing on the Brink. A novel
Peace with Honour
Grandborough. A novel
A Mad Marriage. A novel
Quinze Jours à Ussat-Les-Bains. Itinéraires divers
The Story of Rushen Castle and Rushen Abbey in the Isle of Man [With plates and a map.]
The Physical Geography of the Sea, and its Meteorology
De Paris à Paris à travers les deux mondes. Capitales et grandes villes ... Illustré, etc
Tales of Black-Country Life
A History of Rome from the Earliest Times to the Death of Domitian [With a map.]
Beeton's Historical Romances, Daring Deeds, and Animal Stories. Illustrated, etc
For the Sake of the Family
The History of Great Britain during the reign of Queen Anne
Latin Proverb. (Faith, or the two strong arms: A dilemma. God's Flowers [Poems.])
Jack's Secret, etc [First published in 'Belgravia' with the title 'A Lover's Secret.']
Select British Poets, or new elegant extracts from Chaucer to the present time, with critical remarks. By W. Hazlitt
An Evening Thought [A poem.]
Excursions in Denmark, Norway, and Sweden; including notices of the state of public opinion in those countries, and anecdotes of their Courts
Handbook of Meywar, etc
Mona's Choice. A novel
History up to Date. A concise account of the war of 1898 between the United States and Spain, its causes and the Treaty of Paris
Diary in Turkish and Greek Waters
A Ride through Western Asia ... With illustrations
Summer Moths: a play, etc
Journey to Iceland: and travels in Sweden and Norway ... From the German by C. F. Cooper
Walks and Talks of an American Farmer in England
On Heroes ... Second edition
Descripcion e historia del Castillo de la Aljafería sito extramuros de la ciudad de Zaragoza
Poems
The Battle of the Fly-Laws [In verse.] By Aquarius
My Tour in Palestine and Syria ... Illustrated. With route map and plan of Jerusalem
A List of Wiltshire Sheriffs
Twelve Months in Southern Europe ... With illustrations by the author
Feet of Clay
Australia as it is ... Third edition
The Niger Sources and the borders of the new Sierra Leone Protectorate ... With four full-page illustrations and a map
A Joviall Crew: or, the Merry Beggar. Presented in a comedie, etc
Historical Memorials of Canterbury. The Landing of Augustine, the Murder of Becket, Edward the Black Prince, Becket's Shrine
Vashti; or, 'Until death us do part.' [A novel.]
The Wayside Altar [Poems.]
Hannibal: a drama [With a preface by Arabella Shore.]
William Bathurst [A novel.]
The History of Pendennis; his fortunes and misfortunes, his friends and his greatest enemy. With illustrations on steel and wood by the author
Celebria quaedam anglorum poematia latine reddita
Plain Tales from the Hills
Carulaire des petites communes. Analyses des pièces, publiées par S. Bormans
'I Forbid the Banns.' The story of a comedy which was played seriously ... Fourth edition
Camilla de Solys [A novel.]
The English-Reader ... A choice miscellany ... from the best modern authors ... with ... copious German notes ... Fifth edition, augmented, etc
Ungedruckte Anglo-Normannische Geschichtsquellen. Herausgegeben von F. L
A Journal of Two Years' Travel in Persia, Ceylon, etc
Les Marseillais à Nancy, 1792. Souvenirs de localité, peintures de mœurs
Kestell of Greystone
Estudio de las razas humanas que han ido poblando sucesivamente la Isla de Cuba
Sebastiani's Secret, etc [A novel.]
Songs of Travel and other verses
Parables in verse, etc. etc
Swords and Ploughshares [Poems.]
The City of Fear, and other poems
The Parliamentary History of the County of Sussex; and of the several boroughs and Cinque Ports therein
Such a Lord is Love ... A woman's heart tragedy
Foscari and Julian. Tragedies [in five acts and in verse]
"""


FACT_PROMPT = """Abraham Lincoln died on"""

BAD_FACT_PROMPT = """Abraham Lincoln died on April 14"""


if __name__ == "__main__":
	main()
