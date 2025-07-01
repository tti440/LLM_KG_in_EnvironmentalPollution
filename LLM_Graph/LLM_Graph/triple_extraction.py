import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import json
from tqdm import tqdm
from typing import List
import os

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto", torch_dtype="auto")
if not os.path.exists("triples/"):
	os.makedirs("triples/")
def format_chatml_prompt(system_prompt, user_prompt):
	return (
		f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
		f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
		f"<|im_start|>assistant\n"
	)

def extract_triples(context_text):
	system_msg = (
	"You are a network ontology graph maker who extracts scientific concepts and their relations from a given context, using category theory."
	)

	user_msg = f"""
	  You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory.

	  You are provided with a context chunk (delimited by triple backticks).
	  Your task is to extract the ontology of terms mentioned in the given context. hese terms should represent the key concepts as per the context, including well-defined and widely used names in the field of environmental pollution.

	  Format your output as a list of JSON. Each element of the list contains a pair of terms and the relation between them, like the following:

		"node_1": "A concept from extracted ontology"
		"node_2": "A related concept from extracted ontology"
		"edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"

	  ---

	  Example context:
	  ```Silk is a strong natural fiber used to catch prey in a web. Beta-sheets control its strength.```

	  Example output:
	  [
		{{ "node_1": "spider silk", "node_2": "fiber", "edge": "is" }},
		{{ "node_1": "beta-sheets", "node_2": "strength", "edge": "control" }},
		{{ "node_1": "silk", "node_2": "prey", "edge": "catches" }}
	  ]

	  ---

	  Now extract approximately 10 ontology triples from the following context:
	  ```{context_text}```
	  """
	prompt = format_chatml_prompt(system_msg, user_msg)

	# Tokenize and run inference
	inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
	with torch.no_grad():
		output = model.generate(
			**inputs,
			max_new_tokens=1024,
			temperature=0.2,
			top_p=0.95,
			top_k=50,
			repetition_penalty=1.1,
			pad_token_id=tokenizer.eos_token_id,
			eos_token_id=tokenizer.eos_token_id,
			do_sample=True
		)
	response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
	cleaned_response = response.replace("<|im_end|>", "").replace("</|im_end|>", "").strip()
	triples_list=[]
	index = cleaned_response.find("[")
	while index != -1:
		cleaned_response = cleaned_response[index:]
		end_index = cleaned_response.find("]")
		if end_index != -1:
				if cleaned_response[end_index-2] == ",":
					triples = cleaned_response[:end_index-2] + "]"
				else:
					triples = cleaned_response[:end_index] + "]"
				try:
					triples = json.loads(triples)
					triples_list.extend(triples)
				except Exception as e:
					print("Failed to parse triples:", e)
					print("Raw output:\n", cleaned_response)
				cleaned_response = cleaned_response[end_index+1:]
				index = cleaned_response.find("[")
		else:
				end_index = cleaned_response.rfind("}")
				if index != -1:
					cleaned_response = cleaned_response[:index+1] + "]"
					try:
						triples = json.loads(cleaned_response)
						triples_list.extend(triples)
						break
					except Exception as e:
						print("Failed to parse triples:", e)
						print("Raw output:\n", cleaned_response)
						break
	if len(triples_list) == 0:
		print(f"Error: No triples found in response: {cleaned_response}")
		return []

	user_msg2 = f"""
	  Read this context: 
	  ```{context_text}```

	  Read this ontology:
	  ```{triples_list}```
	  
	  Improve the ontology by renaming nodes so that they have consistent labels that are widely used in the field of environmental science or related fields such as environmental biology, chemistry or pollution.
	  """
	prompt = format_chatml_prompt(system_msg, user_msg2)

	# Tokenize and run inference
	inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
	with torch.no_grad():
		output = model.generate(
			**inputs,
			max_new_tokens=1024,
			temperature=0.2,
			top_p=0.95,
			top_k=50,
			repetition_penalty=1.1,
			pad_token_id=tokenizer.eos_token_id,
			eos_token_id=tokenizer.eos_token_id,
			do_sample=True
		)
	response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
	cleaned_response = response.replace("<|im_end|>", "").replace("</|im_end|>", "").strip()
	triples_list=[]
	index = cleaned_response.find("[")
	while index != -1:
		cleaned_response = cleaned_response[index:]
		end_index = cleaned_response.find("]")
		if end_index != -1:
				if cleaned_response[end_index-2] == ",":
					triples = cleaned_response[:end_index-2] + "]"
				else:
					triples = cleaned_response[:end_index] + "]"
				try:
					triples = json.loads(triples)
					triples_list.extend(triples)
				except Exception as e:
					print("Failed to parse triples:", e)
					print("Raw output:\n", cleaned_response)
				cleaned_response = cleaned_response[end_index+1:]
				index = cleaned_response.find("[")
		else:
				end_index = cleaned_response.rfind("}")
				if index != -1:
					cleaned_response = cleaned_response[:index+1] + "]"
					try:
						triples = json.loads(cleaned_response)
						triples_list.extend(triples)
						break
					except Exception as e:
						print("Failed to parse triples:", e)
						print("Raw output:\n", cleaned_response)
						break
	if len(triples_list) == 0:
		print(f"Error: No triples found in response: {cleaned_response}")
		return []

	return triples_list

def process_zephyr(mistral_result:List, filename:str, replace:bool=False):
	print("Start Triple Extraction with Zephyr...")
	data= mistral_result
	triples = []
	for entry in tqdm(data, desc=f"Processing: "):
		context_chunk = entry["points"]
		triple = extract_triples(context_chunk)
		triples.extend(triple)
		context_chunk = entry["summary"]
		triple = extract_triples(context_chunk)
		triples.extend(triple)
	print(f"Extracted {len(triples)} triples from {len(data)} entries.")
	_, dir_2nd = filename.split("/")
	save_path = os.path.join("triples",dir_2nd.replace(".jsonl", "_triples.jsonl"))
	if replace or not (os.path.exists(save_path)):
		with open(save_path, "w") as f:
			for triple in triples:
				f.write(json.dumps(triple) + "\n")
		print(f"Triples saved to {save_path}")
	elif not replace and os.path.exists(save_path):
		print(f"File {save_path} already exists. Use --replace to overwrite.")
  
