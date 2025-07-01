from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import string
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import os
import pickle
import networkx as nx
import json
import spacy
import pandas as pd
from pyvis.network import Network

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def visualize_subgraph_pyvis(G_sub, output_html):
	net = Network(height="800px", width="100%")
	net.from_nx(G_sub)
	net.force_atlas_2based()
	net.show_buttons()
	net.show(output_html)
 
def merge_nodes(G: nx.DiGraph, threshold=0.9, batch_size=500, embedding_file=None, save_graph=False):
	"""
	Merge nodes in the graph based on cosine similarity of their embeddings.
	
	Args:
		G (networkx.Graph): The input graph with node embeddings.
		threshold (float): The cosine similarity threshold for merging nodes.
		batch_size (int): The number of nodes to process in each batch.
		embedding_file (str): Path to a file containing node embeddings. Format needs to be .pkl If None, embeddings will be generated.
	Returns:
		networkx.Graph: The graph with merged nodes.
	"""
	if embedding_file is None:
		node_list = list(G.nodes)
		embeddings = model.encode(node_list, device="cuda", normalize_embeddings=True)
		embeddings_array={}
		for i in range(len(node_list)):
			embeddings_array[node_list[i]]=embeddings[i]
		with open("node_embeddings.pkl", "wb") as f:
			pickle.dump(embeddings_array, f)
		embedding_file = "node_embeddings.pkl"
	assert ".pkl" in embedding_file, "Embedding file must be a .pkl file"
	
	embeddings = pickle.load(open(embedding_file, "rb"))
	nodes = list(embeddings.keys())
	embeddings_array = np.array([embeddings[node] for node in nodes])
	node_mapping = {}
	merged_nodes = set()
	nodes_to_recalculate = set()

	for batch_start in tqdm(range(0, len(nodes), batch_size), desc="Processing in batches"):
		batch_end = min(batch_start + batch_size, len(nodes))
		batch_nodes = nodes[batch_start:batch_end]
		batch_embeddings = embeddings_array[batch_start:batch_end]

		similarity_matrix = cosine_similarity(batch_embeddings, embeddings_array)

		for i, node in enumerate(batch_nodes):
			if node in merged_nodes:
				continue
			sim_scores = similarity_matrix[i]
			similar_indices = np.where(sim_scores >= threshold)[0]
			similar_nodes = [
				nodes[j] for j in similar_indices
				if nodes[j] != node and nodes[j] not in merged_nodes
			]
			if not similar_nodes:
				continue
			all_candidates = similar_nodes + [node]
			rep_node = max(all_candidates, key=lambda x: G.degree[x])
			for other_node in similar_nodes:
				if other_node != rep_node:
					node_mapping[other_node] = rep_node
					merged_nodes.add(other_node)
					nodes_to_recalculate.add(rep_node)

	# Rebuild the graph with merged nodes
	new_graph = nx.relabel_nodes(G, node_mapping, copy=True)
	if save_graph:
		nx.write_graphml(new_graph, "node_merged_graph.graphml")
	with open("node_mapping(0.9).json", "w") as f:
		json.dump(node_mapping, f, indent=4)
	return new_graph

def normalize_label_preserve_be(label):
	doc = nlp(label.lower())
	lemmas = []
	aux_seen = False
	
	for token in doc:
		#print(token.text, token.dep_, token.lemma_, token.pos_)
		if token.dep_ == "aux" or token.lemma_ == "be":
			if not aux_seen:
				lemmas.append("be")
				aux_seen = True
		elif token.pos_ in {"VERB", "NOUN", "ADJ"} and token.lemma_ != "be":
			if "be" in lemmas:
				lemmas.append(token.text)
			else:
				lemmas.append(token.lemma_)
		# optionally include prepositions and important modifiers
		elif token.pos_ in {"ADP"}:
			lemmas.append(token.lemma_)

	return " ".join(lemmas)


def normalize_label_preserve_voice(label):
	doc = nlp(label.lower())
	lemmas = []
	aux_seen = False
	is_passive = False

	for token in doc:
		# Detect passive voice with auxpass + nsubjpass etc.
		if token.dep_ in {"auxpass", "nsubjpass"}:
			is_passive = True
		if token.dep_ == "aux" or token.lemma_ == "be":
			if not aux_seen:
				lemmas.append("be")
				aux_seen = True
		elif token.pos_ in {"VERB", "NOUN", "ADJ"} and token.lemma_ != "be":
			if "be" in lemmas:
				lemmas.append(token.text)
			else:
				lemmas.append(token.lemma_)
		elif token.pos_ in {"ADP"}:
			lemmas.append(token.lemma_)

	normalized = " ".join(lemmas)
	if is_passive:
		normalized = "[PASSIVE] " + normalized
	else:
		normalized = "[ACTIVE] " + normalized
	return normalized

def merge_edge(G: nx.DiGraph, threshold=0.9, batch_size=500, embedding_file=None, save_graph=False):
	'''
	Merge edges in the graph based on cosine similarity of their embeddings.
	Args:
		G (networkx.Graph): The input graph with edge embeddings.
		threshold (float): The cosine similarity threshold for merging edges.
		batch_size (int): The number of edges to process in each batch.
		embedding_file (str): Path to a file containing edge embeddings. Format needs to be .pkl If None, embeddings will be generated.
	Returns:
		networkx.Graph: The graph with merged edges.
	'''
	
	if embedding_file is None:
		edge_list = set()
		for u, v, data in G.edges(data=True):
			edge_list.add(data['label'])
		edge_list = list(edge_list)
		edge_list = [normalize_label_preserve_be(edge) for edge in edge_list]
		edge_list = set(edge_list)
		edge_list = list(edge_list)
		edge_embeddings = model.encode(edge_list, device="cuda", normalize_embeddings=True)
		edge_mapping = {}
		for i in range(len(edge_list)):
			edge_mapping[edge_list[i]] = edge_embeddings[i]
		try:
			edge_mapping.pop("")
		except:
			pass
		with open("edge_embeddings.pkl", "wb") as f:
			pickle.dump(edge_mapping, f)
		embedding_file = "edge_embeddings.pkl"
	assert ".pkl" in embedding_file, "Embedding file must be a .pkl file"
	edge_embeddings = pickle.load(open("edge_embeddings.pkl", "rb"))
	G_new = G.copy()
	if "" in edge_embeddings:
		edge_embeddings.pop("")
	
	normalized_map = {}
	for label in list(edge_embeddings.keys()):
		if not label:
			continue
		norm = normalize_label_preserve_voice(label)
		normalized_map[label] = norm
		if norm:
			normalized_map[label] = norm
		else:
			normalized_map[label] = label

	
	merged_embeddings = {}
	for orig_label, norm_label in normalized_map.items():
		if norm_label not in merged_embeddings:
			merged_embeddings[norm_label] = []
		merged_embeddings[norm_label].append(edge_embeddings[orig_label])

	
	label_embeddings = []
	edge_labels = []
	for label, vectors in merged_embeddings.items():
		avg_vec = np.mean(np.array(vectors), axis=0)
		edge_labels.append(label)
		label_embeddings.append(avg_vec)
	label_embeddings = np.array(label_embeddings)

	
	label_to_index = {label: i for i, label in enumerate(edge_labels)}
	index_to_label = {i: label for label, i in label_to_index.items()}

	merged_labels = set()
	label_mapping = {}
	for batch_start in tqdm(range(0, len(edge_labels), batch_size), desc="Merging edge labels"):
		batch_end = min(batch_start + batch_size, len(edge_labels))
		batch_labels = edge_labels[batch_start:batch_end]
		batch_embeddings = label_embeddings[batch_start:batch_end]

		similarity_matrix = cosine_similarity(batch_embeddings, label_embeddings)

		for i, label in enumerate(batch_labels):
			if label in merged_labels:
				continue
			sim_scores = similarity_matrix[i]
			similar_indices = np.where(sim_scores >= threshold)[0]
			similar_labels = [
				edge_labels[j] for j in similar_indices
				if edge_labels[j] != label and edge_labels[j] not in merged_labels
			]
			if not similar_labels:
				continue
			all_candidates = similar_labels + [label]
			if label.startswith("[ACTIVE] "):
				all_candidates = [l for l in all_candidates if l.startswith("[ACTIVE] ")]
			elif label.startswith("[PASSIVE] "):
				all_candidates = [l for l in all_candidates if l.startswith("[PASSIVE] ")]
			rep_label = min(all_candidates, key=len)
			label_mapping[label] = rep_label
			merged_labels.add(label)
		
	
	for u, v, data in G_new.edges(data=True):
		orig_label = data.get("label")
		norm_label = normalized_map.get(orig_label)
		if norm_label in label_mapping:
			mapped_label = label_mapping[norm_label]
			#remove [PASSIVe] or [ACTIVE] if present
			if mapped_label.startswith("[PASSIVE] "):
				mapped_label = mapped_label[len("[PASSIVE] "):]
			elif mapped_label.startswith("[ACTIVE] "):
				mapped_label = mapped_label[len("[ACTIVE] "):]
			data["label"] = mapped_label
		elif norm_label:  # fallback to normalized label if no mapping
			#remove [PASSIVe] or [ACTIVE] if present
			if norm_label.startswith("[PASSIVE] "):
				norm_label = norm_label[len("[PASSIVE] "):]
			elif norm_label.startswith("[ACTIVE] "):
				norm_label = norm_label[len("[ACTIVE] "):]
			data["label"] = norm_label

	if save_graph:
		nx.write_graphml(G_new, "edge_merged_graph.graphml")
	with open("label_mapping(0.9).json", "w") as f:
		json.dump(label_mapping, f, indent=4)
	with open("normalized_map.json", "w") as f:
		json.dump(normalized_map, f, indent=4)
	return G_new

def check_duplicates(G: nx.DiGraph):
	"""
	Check for duplicate nodes in the graph based on their labels.
	
	Args:
		G (networkx.Graph): The input graph.
	Returns:
		list: A list of tuples containing the duplicate nodes.
	"""
	duplicates = []
	triples = set()
	for u, v, data in G.edges(data=True):
		edge = data.get("label")
		triple = (u, edge, v)
		if triple in triples:
			duplicates.append((u, v, edge))
			print(f"Duplicate found: {u} - {v} with edge '{edge}'")
		else:
			triples.add(triple)
	G_new = nx.DiGraph()
	for u, v, data in G.edges(data=True):
		edge = data.get("label")
		if (u, v, edge) not in duplicates:
			G_new.add_edge(u, v, label=edge)
	return G_new

def is_empty_or_punct(s):
	return not s.strip() or all(c in string.punctuation for c in s.strip())

def data_preprocess():
	path = "data/Industrial_Reporting_Facilities_relevant_contribution.xlsx"
	df = pd.read_excel(path, sheet_name="Facilities_relevant_contributio")
	erptr_path = "data/erptr_industry_code.json"
	erptr = json.load(open(erptr_path, "r", encoding="utf-8"))
	column = df["EPRTR_AnnexI_MainActivity"]
	indistry_names=[]
	for entry in column:
		index = entry.find(")")
		industry_name = entry[:index+1].strip()  # Extract the part before the first parenthesis
		indistry_names.append(erptr[industry_name]["main_activity"])
	df["EPRTR_AnnexI_MainActivity"]= indistry_names
	df.to_excel("data/Updated_Industrial_Reporting_Facilities_relevant_contribution.xlsx", index=False, sheet_name="Facilities_relevant_contributio")
 
def get_term_mapping(G: nx.DiGraph, df: pd.DataFrame):
	terms = set()
	for term in df["countryName"]:
		terms.add(term.lower())
	for term in df["Pollutant"]:
		terms.add(term.lower())
	for term in df["EPRTR_AnnexI_MainActivity"]:
		terms.add(term.lower())
	nodes_list = list(G.nodes)
	node_embedding = model.encode(nodes_list, device="cuda", normalize_embeddings=True)
	terms_in_graph = defaultdict(str)
	for search_term in terms:
		embedding = model.encode([search_term], device="cuda", normalize_embeddings=True)
		similarity = cosine_similarity(embedding, node_embedding)[0]
		#take highest similarity node
		similarity_index = similarity.argmax()
		terms_in_graph[search_term] = nodes_list[similarity_index]
	with open("terms_in_graph.json", "w") as f:
		json.dump(terms_in_graph, f, indent=4)
  
def map_data(G: nx.DiGraph, df: pd.DataFrame, verbose:bool):
	"""
	Map data from the DataFrame to the graph nodes.
	
	Args:
		G (networkx.Graph): The input graph.
		df (pd.DataFrame): The DataFrame containing mapping data.
	Returns:
		graph (networkx.Graph): The graph with mapped data.
	"""
	node_list = list(G.nodes)
	node_embeddings_array = model.encode(node_list, device="cuda", normalize_embeddings=True)
	terms_in_graph = json.load(open("terms_in_graph.json", "r", encoding="utf-8"))
	for index, row in df.iterrows():
		try:
			# Normalize inputs
			raw_country = row["countryName"].lower()
			raw_pollutant = row["Pollutant"].lower()
			raw_activity = row["EPRTR_AnnexI_MainActivity"].lower()

			country = terms_in_graph[raw_country]
			pollutant = terms_in_graph[raw_pollutant]
			activity = terms_in_graph[raw_activity]
			emission_data = row[["2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]].to_dict()
			
			for year, value in emission_data.items():
				if pd.notna(value) and value > 0:
					intermediate_node = f"{country}_{activity}_{year}"
					
					# Add intermediate node if not already present
					if not G.has_node(intermediate_node):
						G.add_node(intermediate_node, type="emission_event", year=year, amount=float(value), activity=activity)
					G.add_node(country, type="country")
					G.add_node(pollutant, type="pollutant")
					G.add_node(activity, type="activity")
					
					# Add relations
					G.add_edge(country, intermediate_node, label="has_emission_event")
					G.add_edge(intermediate_node, pollutant, label="emits", weight=float(value), year = year, activity=activity)  # add weight for value
					
					G.add_edge(intermediate_node, f"{value:.4f}", label="has_amount")  # keep value stringified
					G.add_edge(intermediate_node, activity, label="has_activity")
					G.add_edge(intermediate_node, year, label="has_year")

		except Exception as e:
			print(f"Skipping row {index} due to error: {e}")

	pollutants = df.Pollutant.unique()
	for p in pollutants:
		pollutant = terms_in_graph[p.lower()]   
		pollutant_embed = model.encode([pollutant], device="cuda", normalize_embeddings=True)
		sim_matrix = cosine_similarity(pollutant_embed, node_embeddings_array)[0]
		sim_bool = sim_matrix > 0.85
		for i, similar in enumerate(sim_bool):
			if similar:
					G.add_edge(pollutant, node_list[i], label="similar_to", weight=float(sim_matrix[i]))
					G.add_edge(node_list[i], pollutant, label="similar_to", weight=float(sim_matrix[i]))
					G.add_node(node_list[i], type="pollutant")
					if verbose:
						print(f"Adding edge from {pollutant} to {node_list[i]} with similarity {sim_matrix[i]}")	
	for raw_pollutant in pollutants:
		pollutant = terms_in_graph[raw_pollutant.lower()]
		total = df[df.Pollutant == raw_pollutant].groupby("countryName").sum()[["2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]].sum(axis=1)
		for country, value in total.items():
			if pd.notna(value) and value > 0:
				country_node = terms_in_graph[country.lower()]
				G.add_edge(country_node, pollutant, label="emits", weight=float(value))
	nx.write_graphml(G, "final_mapped_cleaned_graph.graphml")
	return G

def get_final_embeddings(G):
	node_list = list(G.nodes)
	node_embeddings = model.encode(node_list, device="cuda", normalize_embeddings=True)
	node_embeddings_array = {node: node_embeddings[i] for i, node in enumerate(node_list)}
	with open("final_node_embeddings.pkl", "wb") as f:
		pickle.dump(node_embeddings_array, f)
	edge_list = set()
	for u, v, data in G.edges(data=True):
		edge_list.add(data['label'])
	edge_list = list(edge_list)
	edge_list = [normalize_label_preserve_be(edge) for edge in edge_list]
	edge_list = set(edge_list)
	edge_list = list(edge_list)
	edge_embeddings = model.encode(edge_list, device="cuda", normalize_embeddings=True)
	edge_mapping = {}
	for i in range(len(edge_list)):
		edge_mapping[edge_list[i]] = edge_embeddings[i]
	with open("final_edge_embeddings.pkl", "wb") as f:
		pickle.dump(edge_mapping, f)
