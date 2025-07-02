from Bio import Entrez
import time
import json
from lxml import etree
from nltk.tokenize import sent_tokenize
import spacy
import re
import os
from dotenv import load_dotenv
import os

load_dotenv()  # load from .env

email = os.getenv("your@email.com")

Entrez.email = email  # Always include this
def normalize_whitespace(text: str) -> str:
	# Replace multiple spaces, tabs, and newlines with a single space
	return re.sub(r'\s+', ' ', text).strip()
def decode_unicode_escapes(text: str) -> str:
	return text.encode('utf-8').decode('unicode_escape')
def fix_utf8_garbage(text):
	try:
		return bytes(text, "utf-8").decode("utf-8")
	except Exception:
		return text  # fallback if decoding fails
	
def remove_tags(element, tags_to_remove):
	"""Remove unwanted math or display blocks from the XML."""
	for tag in tags_to_remove:
		for match in element.findall(f'.//{tag}', namespaces=element.nsmap):
			parent = match.getparent()
			if parent is not None:
				parent.remove(match)

def parse_pmc_xml(k, xml_str, chunk_size=750):
	if type(xml_str) == str:
		root = etree.fromstring(xml_str.encode('utf-8'))
	else:
		root = xml_str.getroot()
	ns = {'ns': 'http://www.ncbi.nlm.nih.gov/JATS1'}  # JATS is PMC's XML format
	body = root.find('.//body')

	all_text = []
	try:
		remove_tags(body, ['tex-math',
		'inline-formula',
		'disp-formula',
		'alternatives',
		'graphic'])
		for sec in body.iterfind('.//sec'):
			title = sec.findtext('title') or ""
			paragraphs = ["".join(p.itertext()).strip() for p in sec.findall('p') if p is not None]
			sec_text = title + "\n" + "\n".join(paragraphs)
			all_text.append(sec_text)
	except Exception as e:
		print(f"Error processing section: {e}")
		print(f"Id: {k}")

	full_text = "\n\n".join(all_text)
	full_text = full_text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("[", "")
	# Split into chunks of ~chunk_size words
	tokenizer = spacy.load("en_core_web_sm")
	doc = tokenizer(full_text)
	sentences = [sent.text for sent in doc.sents]
	chunks, current, count = [], [], 0
	for s in sentences:
		s = normalize_whitespace(s)
		w = len(s.split())
		if count + w > chunk_size:
			chunks.append(" ".join(current))
			current, count = [], 0
		current.append(s)
		count += w
	if current:
		chunks.append(" ".join(current))

	return chunks

def fetch_pubmed_literature(query, max_results=100, engine_type="pubmed"):
	'''
	fetch literatre from PubMed or PMC based on the query.
	PMC is used for full text articles, while PubMed is used for abstracts.
	Args:
		query (str): The search term to query PubMed or PMC.
  		max_results (int): The maximum number of results to return.
  		engine_type (str): The type of engine to use, either "pubmed" for
	Returns:
	A list of dictionaries containing PubMed IDs and either abstracts or full texts.
	'''
	if engine_type == "pubmed":
		search_handle = Entrez.esearch(db="pmc", term=query, retmax=max_results)
		search_results = Entrez.read(search_handle)
		id_list = search_results["IdList"]
		abstracts = []
		for pubmed_id in id_list:
			handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="abstract", retmode="text")
			abstract_text = handle.read()
			abstracts.append({
					"pubmed_id": pubmed_id,
					"abstract": abstract_text
				})
		return abstracts
	else:
		search_handle = Entrez.esearch(db="pmc", term=query, retmax=max_results)
		search_results = Entrez.read(search_handle)
		id_list = search_results["IdList"]
		full_texts = []
		for pubmed_id in id_list:
			try:
				handle = Entrez.efetch(db="pmc", id=pubmed_id, rettype="full", retmode="xml")
			except:
				print(f"Error fetching full text for ID {pubmed_id}")
				continue
			corpus = handle.read()
			xml_str = corpus.decode("utf-8")
			full_texts.append({
					"pubmed_id": pubmed_id,
					"text": xml_str
				})
			time.sleep(0.3)
		return full_texts

def text_collection(queries):
	pmc_results = {}
	for query in queries:
		corpus = fetch_pubmed_literature(
			query,
			max_results=100,
			engine_type="pmc"
		)
		for entry in corpus:
			if entry['pubmed_id'] not in pmc_results:
				pmc_results[entry['pubmed_id']] = entry['text']
		time.sleep(1.0)
	if not os.path.exists("corpus_text_json/pmc_corpus.jsonl"):
		with open("corpus_text_json/pmc_corpus.jsonl", "w") as f:
			for k, v in pmc_results.items():
				try:
					chunks = parse_pmc_xml(k, v, chunk_size=750)
				except:
					print(f"Error parsing PMC XML for ID {k}")
					continue
				f.write(json.dumps({"pubmed_id": k, "text": chunks}) + "\n")
		f.close()
  
from typing import List
def fulltext_2_chunks(filename:List, output_file:str=None, chunk_size:int=750):
	'''
	Convert full text corpus to chunks.
	Args:
		filename (list): The path to the corpus files.
		output_file (str): The path to the output jsonl file.
		chunk_size (int): The maximum size of each chunk in words.

	'''
	if output_file is None:
		name = filename[0].split(".")[0]
		output_file = f"{name}.jsonl"
	full_text = []
	for f in filename:
		assert os.path.exists(f), f"File {f} does not exist."
		assert f.endswith(".txt"), f"File {f} is not a txt file."
		with open(f, "r") as file:
			for line in file:
				line = line.strip()
				if line:
					full_text.append(line)
	
	chunks = []
	id = 0
	tokenizer = spacy.load("en_core_web_sm")
	with open(f"corpus_text_json/{output_file}", "w", encoding="utf-8") as out:
		full_text = ' '.join(full_text)
		doc = tokenizer(full_text)
		sentences = [sent.text for sent in doc.sents]
		chunks, current, count = [], [], 0
		chunk_size = 750
		for s in sentences:
			w = len(s.split())
			if count + w > chunk_size:
				chunks.append(" ".join(current))
				current, count = [], 0
			current.append(s)
			count += w
		if current:
			chunks.append(" ".join(current))
		out.write(json.dumps({"id": f"{id}", "text":chunks}) + "\n")
		id += 1
		out.close()
	print(f"Chunks saved to corpus_text_json/{output_file}")