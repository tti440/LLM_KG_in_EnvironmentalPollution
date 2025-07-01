import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm
import gc
import os


torch.cuda.empty_cache()
gc.collect()
# === Load model and tokenizer ===
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
set_seed(42)
if not os.path.exists("mistral_output/"):
	os.makedirs("mistral_output/")
 
# === Format prompt in ChatML style ===
def format_chatml_prompt(system_prompt, user_prompt):
	return (
		f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
 		f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
		f"<|im_start|>assistant\n"
	)

# === Generate LLM output ===
def generate_response(prompt, max_tokens=1024):
	inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=max_tokens,
			temperature=0.1,
			top_p=0.95,
			top_k=50,
			repetition_penalty=1.1,
			eos_token_id=tokenizer.eos_token_id,
			do_sample=True
		)
	response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
	response = response.replace("<|im_end|>", "").strip()
	response = response.replace("</|im_end|>", "").strip()
	del inputs, outputs
	torch.cuda.empty_cache()
	gc.collect()
	return response.strip()

# === Prompt functions ===
def get_summary(text_chunk):
	system_prompt = "You are a scientific assistant. You respond with a concise scientific summary, including reasoning. You never use names or references."
	user_prompt = f'In a matter-of-fact voice, rewrite this: """{text_chunk}"""\nThe writing must stand on its own and include relevant background and details.'
	return generate_response(format_chatml_prompt(system_prompt, user_prompt))

def get_bullet_points(summary_text):
	system_prompt = "You are a scientific assistant. You respond with scientific reasoning in bullet points. You never use names or references."
	user_prompt = f'Provide a bullet point list of the key facts and reasoning in """{summary_text}""". Think step by step. Do not use citations or names.'
	return generate_response(format_chatml_prompt(system_prompt, user_prompt))

def get_title(summary_text):
	system_prompt = "You are a scientist writing a paper. Do not use citations."
	user_prompt = f'Provide a one-sentence title of this text: """{summary_text}""". Make sure the title can be understood without additional context.'
	return generate_response(format_chatml_prompt(system_prompt, user_prompt))

def process_mistral(file:str, save_to_file:bool=False):
	'''
	Process a JSONL file with Mistral model.
	Args:
		file (str): Path to the input JSONL file.
	Returns:
		None
	'''
	with open(file, "r", encoding="utf-8") as f1:
		data = [json.loads(line) for line in f1]
	mistral_output = []
	print(f"Processing {len(data)} entries from {file} with Mistral model...")
	for item in data:
		pubmed_id = item["pubmed_id"]
		paragraphs = item["text"]
		for idx, paragraph in enumerate(tqdm(paragraphs, desc=f"Processing {pubmed_id}", leave=False)):
			try:
				summary = get_summary(paragraph)
				points = get_bullet_points(summary)
				title = get_title(summary)
				uid = f"{pubmed_id}_{idx}"
				mistral_output.append({
					"pubmed_id": uid,
					"summary": summary,
					"points": points,
					"title": title
				})
			except Exception as e:
				print(f"[Error] Skipped {pubmed_id}_{idx} due to: {e}")
				continue
	print("Processing complete.")
	if save_to_file:
		_, dir_2nd = file.split("/")
		out_file = os.path.join("mistral_output/",dir_2nd.replace(".jsonl", "_mistral_processed.jsonl"))
		with open(out_file, "w", encoding="utf-8") as out:
			for item in mistral_output:
				out.write(json.dumps(item) + "\n")
	return mistral_output