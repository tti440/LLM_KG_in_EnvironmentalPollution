from extraction_process import text_collection
from summarisation_mistral import process_mistral
from triple_extraction import process_zephyr
import os
import json
import networkx as nx
from construction_utils import merge_edge, merge_nodes, check_duplicates, is_empty_or_punct, \
	data_preprocess, get_term_mapping, map_data, get_final_embeddings
from collections import Counter
import pandas as pd


def triples_generation():
	queries = [
	'("industrial pollution" OR "industrial emissions") AND ("chemical substances" OR "toxic waste") AND (Europe OR European)',
	'"emissions from manufacturing" AND ("pollutants" OR "hazardous substances") AND (Germany OR France OR Netherlands)',
	'"chemical plant" AND ("emissions" OR "pollution") AND ("waste management" OR "discharge") AND Europe',
	'"air pollution" AND ("coal plants" OR "steel industry") AND (Poland OR Germany) AND "environmental impact"',
	'("marine pollution" OR "water pollution") AND ("industrial discharge") AND (Spain OR Italy OR UK)',
	'("heavy metals" OR "lead" OR "cadmium" OR "mercury") AND ("industrial source" OR "manufacturing site") AND Europe'
	'("plastic production" OR "textile industry") AND ("chemical leakage" OR "pollutants") AND Europe',
	'("pesticide production" OR "fertilizer industry") AND ("chemical release" OR "air pollution") AND EUROPE',
	'("electronics manufacturing") AND ("e-waste" OR "toxic byproducts") AND ("environmental exposure" OR "soil contamination")',
	'("EU regulation" OR "European law") AND ("industrial emissions" OR "air quality directive") AND ("compliance" OR "impact")',
	'"REACH regulation" AND ("chemical safety" OR "substance restriction") AND ("industrial compliance")',
	'("environmental policy" OR "industrial pollution control") AND ("EU directives" OR "European Commission")',
	'("BREF documents" OR "Best Available Techniques") AND ("emissions limit values" OR "industrial compliance")',
	'"PFAS pollution" AND ("industrial source" OR "chemical production") AND Europe',
	'"NOx emissions" AND ("combustion plants" OR "power industry") AND ("air quality") AND Europe',
	'"microplastics" AND ("wastewater treatment" OR "industrial discharge") AND ("environmental monitoring")'
	]
	# Results are saved in corpus_text_json/pmc_corpus.jsonl
	# If the file already exists, it will not be overwritten.
	if not os.path.exists("corpus_text_json/pmc_corpus.jsonl"):
		text_collection(queries)
	corpus_files = os.listdir("corpus_text_json/")
	for file in corpus_files:
		if file.endswith(".jsonl"):
			file_path = os.path.join("corpus_text_json", file)
			# Change if you want to save the processed file
			#mistral_output=process_mistral(file_path, save_to_file=False)
			#process_zephyr(mistral_output, file_path, replace=False)
   
	triple_files = os.listdir("triples/")
	triples = []
	for file in triple_files:
		if file.endswith("_triples.jsonl"):
			file_path = os.path.join("triples", file)
			with open(file_path, "r", encoding="utf-8") as f:
				triples.extend([json.loads(line) for line in f])
	
	if not os.path.exists("all_triples.jsonl"):
		with open("all_triples.jsonl", "w", encoding="utf-8") as f:
			for triple in triples:
				f.write(json.dumps(triple) + "\n")
	
def graph_construction(node_threshold=0.9, node_embedding_file=None, edge_threshold=0.9, edge_embedding_file=None, verbose=False):
	triples= []
	
	with open("all_triples.jsonl", "r", encoding="utf-8") as f:
		for line in f:
			try:
				triple = json.loads(line)
				triples.append(triple)
			except json.JSONDecodeError as e:
				print(f"Error decoding JSON: {e}")
				continue
				
	triples_set = set()	
	for triple in triples:
		try:
			if "node1" in triple and "relation" in triple and "node2" in triple:
				triple_tuple = (triple['node1'], triple['relation'], triple['node2'])
				triples_set.add(triple_tuple)
			elif "node_1" in triple and "relation" in triple and "node_2" in triple:
				triple_tuple = (triple['node_1'], triple['relation'], triple['node_2'])
				triples_set.add(triple_tuple)
		except:
			if verbose:
				print(f"Skipping triple due to missing keys or wrong formats: {triple}")
	triples=[]
	for triple in triples_set:
		triples.append({
			"node1": triple[0],
			"relation": triple[1],
			"node2": triple[2]
		})
	G = nx.DiGraph()
	for triple in triples:
		try:
			n1 = triple["node1"].lower()
			n2 = triple["node2"].lower()
			edge = triple["relation"].lower().replace("_", " ")
			G.add_edge(n1, n2, label=edge)
		except:
			if verbose:
				print(f"Key error: in triple {triple}")
	new_G = merge_nodes(G, threshold=node_threshold, batch_size=500, embedding_file=node_embedding_file, save_graph=False)
	new_G = merge_edge(new_G, threshold=edge_threshold, batch_size=500, embedding_file=edge_embedding_file, save_graph=False)
	new_G = check_duplicates(new_G)
	for comp in list(nx.weakly_connected_components(new_G)):
		if len(comp) < 10:
			new_G.remove_nodes_from(comp)
	G = new_G.copy()
	edges_to_remove = [
		(u, v) for u, v, d in G.edges(data=True)
		if is_empty_or_punct(d.get("label", ""))
	]
	G.remove_edges_from(edges_to_remove)
	nodes_to_remove = [n for n in G.nodes if is_empty_or_punct(n)]
	G.remove_nodes_from(nodes_to_remove)
	isolated = list(nx.isolates(G))
	G.remove_nodes_from(isolated)
	for comp in list(nx.weakly_connected_components(G)):
		if len(comp) < 10:
			G.remove_nodes_from(comp)
	print(f"Cleaned graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
	nx.write_graphml(G, "final_graph_cleaned.graphml")
	if verbose:
		comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
		print(f"Largest component size: {len(comps[0])}")
		print(f"Top 5 component sizes: {[len(c) for c in comps[:5]]}")

		degree_dict = dict(G.degree())
		top_nodes = Counter(degree_dict).most_common(10)
		for node, deg in top_nodes:
			print(f"{node}: degree {deg}")
		label_counts = Counter([d['label'] for _, _, d in G.edges(data=True)])
		for label, count in label_counts.most_common(10):
			print(f"{label}: {count}")
   
def graph_mapping(verbose):
	G = nx.read_graphml("final_graph_cleaned.graphml")
	if not os.path.exists("data/Updated_Industrial_Reporting_Facilities_relevant_contribution.xlsx"):
		data_preprocess()
	df = pd.read_excel("data/Updated_Industrial_Reporting_Facilities_relevant_contribution.xlsx", sheet_name="Facilities_relevant_contributio")
	if not os.path.exists("terms_in_graph.json"):
		get_term_mapping(G, df)
	graph = map_data(G, df, verbose)
	nx.write_graphml(graph, "final_mapped_cleaned_graph.graphml")
	get_final_embeddings(graph)

def pipeline(node_threshold, node_embedding_file, edge_threshold, edge_embedding_file, verbose):
	print("Starting triples generation...")
	triples_generation()
	print("Triples generation complete.")
	
	print("Starting graph construction...")
	graph_construction(node_threshold, node_embedding_file, edge_threshold, edge_embedding_file, verbose)
	print("Graph construction complete.")
	
	print("Starting graph mapping...")
	graph_mapping(verbose)
	print("Graph mapping complete.")

import argparse
if __name__ == "__main__":
	argparse = argparse.ArgumentParser(description="Run the LLM pipeline for environmental pollution data.")
	argparse.add_argument("--node_threshold", type=float, default=0.9, help="Threshold for merging nodes.")
	argparse.add_argument("--node_embedding_file", type=str, default=None, help=".pkl file for node embeddings.")
	argparse.add_argument("--edge_threshold", type=float, default=0.9, help="Threshold for merging edges.")
	argparse.add_argument("--edge_embedding_file", type=str, default=None, help=".pkl file for edge embeddings.")
	argparse.add_argument("--verbose", type = str, default="False", help="Enable verbose output.")
	args = argparse.parse_args()
	if args.verbose == "True":
		args.verbose = True
	elif args.verbose == "False":
		args.verbose = False
	pipeline(args.node_threshold, args.node_embedding_file, args.edge_threshold, args.edge_embedding_file, args.verbose)