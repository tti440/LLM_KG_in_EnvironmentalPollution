import networkx as nx
from pyvis.network import Network
from GraphReasoning import *
from transformers import AutoTokenizer, AutoModel, set_seed
import torch
from openai import OpenAI
import os
import pandas as pd
import json
import nx_cugraph as nxcg
import networkx as nx
import cudf
import cupy as cp
from joblib import Parallel, delayed
import pickle
from matplotlib import pyplot as plt
import math
from tqdm import tqdm
import numpy as np
import cugraph
import powerlaw

os.environ["NX_CUGRAPH_AUTOCONFIG"] = "1"
set_seed(42) 
tokenizer_model = "BAAI/bge-large-en-v1.5"
embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
embedding_model = AutoModel.from_pretrained(tokenizer_model)
client = OpenAI(api_key="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")

if not os.path.exists("output"):
    os.makedirs("output")
    
def visualize_subgraph_pyvis(G_sub, output_html):
	net = Network(height="800px", width="100%", notebook=True, cdn_resources="in_line")
	net.from_nx(G_sub)
	net.force_atlas_2based()
	net.show_buttons()
	net.show(output_html)
	print(f"Interactive graph saved to: output/{output_html}")

def subgraph_extraction(G, target, method, similar_to=True, graphml=True):
	"""
	Extract a subgraph from the main graph based on the target node and method.
	
	Args:
		G (nx.DiGraph): The main graph.
		target (str): The target node to extract the subgraph for.
		method (str): The method to use for extraction, either "1hop" or "2hops"
		similar_to (bool): Whether to include nodes similar to the target node.
	
	"""
	targets = [target]
	if not G.has_node(target):
		raise ValueError(f"Target node '{target}' does not exist in the graph.")
	if similar_to:
		for u,v,edge in G.edges(data=True):
			if edge.get("label") == "similar_to" and (u == target or v == target):
				if u != target:
					targets.append(u)
				if v != target:
					targets.append(v)
	if method == "1hop":
		preds = set()
		succs = set()
		for t in targets:
			preds.update(G.predecessors(t))
			succs.update(G.successors(t))
		subgraph_nodes = preds.union(succs).union(targets)
		G_sub = G.subgraph(subgraph_nodes).copy()
	elif method == "2hops":
		nodes = set()
		for t in targets:
			nodes.add(t)
			nodes.update(G.predecessors(t))
			nodes.update(G.successors(t))
			for pred in G.predecessors(t):
				nodes.update(G.predecessors(pred))
				nodes.update(G.successors(pred))
			for succ in G.successors(t):
				nodes.update(G.predecessors(succ))
				nodes.update(G.successors(succ))
		G_sub = G.subgraph(nodes).copy()
	else:
		raise ValueError("Method must be either '1hop' or '2hops'")
	visualize_subgraph_pyvis(G_sub, f"{target}_{method}_subgraph.html")
	if graphml:
		graphml_file = f"{target}_{method}_subgraph.graphml"
		nx.write_graphml(G_sub, graphml_file)
		print(f"Subgraph saved to: output/{graphml_file}")
 

def generate(system_prompt, prompt, max_tokens=2048, temperature=0.1):
	response = client.chat.completions.create(
		model="gpt-4.1",
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": prompt}
		],
		max_tokens=max_tokens,
		temperature=temperature
	)
	return response.choices[0].message.content

def generate_response(
	keyword_1=None,
	keyword_2=None,
	N_limit=9999,  # The limit for keywords, triplets, etc.
	instruction=None,
	temperature=0.1, 
	prepend=None,  # Prepend text for analysis
	visualize_paths_as_graph=True,  # Whether to visualize paths as a graph
	display_graph=True,  # Whether to display the graph
):
	graph_file = 'final_mapped_cleaned_graph.graphml'
	embedding_file = 'final_node_embeddings.pkl'
	data_dir_output = './GRAPHDATA_OUTPUT/'
	make_dir_if_needed(data_dir_output)
	G = nx.read_graphml(graph_file)
	for u,v, data in G.edges(data=True):
		data["title"] = data.get("label", "")
		G.add_edge(u, v, **data)
	node_embeddings = load_embeddings(embedding_file)
	response, (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML = find_path_and_reason(
	G, 
	node_embeddings,
	embedding_tokenizer, 
	embedding_model, 
	generate, 
	data_dir=data_dir_output,
	verbatim=True,
	include_keywords_as_nodes=False,  # Include keywords in the graph analysis
	keyword_1=keyword_1,  # First keyword to search for
	keyword_2=keyword_2,  # Second keyword to search for
	N_limit=N_limit,  # The limit for keywords, triplets, etc.
	instruction=instruction,  # Instruction for the analysis
	keywords_separator=', ',
	graph_analysis_type='nodes and relations',
	temperature=temperature,  # Temperature for the generation
	inst_prepend='### ',  # Instruction prepend text
	prepend=prepend,  # Prepend text for analysis
	visualize_paths_as_graph=visualize_paths_as_graph,  # Whether to visualize paths as a graph
	display_graph=display_graph,  # Whether to display the graph
	)
	return response, (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML

def ppr_top10(G, target):
	ppr = nx.pagerank(G, alpha=0.85, personalization={target: 1.0})
	if target in ppr:
		ppr.pop(target)
	top10 = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:10]
	return top10

def ppr_pollutants(G):
	df = pd.read_excel("data/Updated_Industrial_Reporting_Facilities_relevant_contribution.xlsx")
	pollutants = list(df.Pollutant.unique())
	terms_in_graph = json.load(open("terms_in_graph.json", "r"))
	resluts = {}
	pollutants_in_graph = [terms_in_graph[p.lower()] for p in pollutants]
	for pollutant in pollutants_in_graph:
		res = ppr_top10(G, pollutant)
		resluts[pollutant] = res
	with open("ppr_pollutants.jsonl", "w") as f:
		for pollutant, res in resluts.items():
			f.write(json.dumps({"pollutant": pollutant, "top10": res}) + "\n")
	return resluts

def ppr_country(G):
	df = pd.read_excel("data/Updated_Industrial_Reporting_Facilities_relevant_contribution.xlsx")
	countries = list(df.countryName.unique())
	countries = [c.lower() for c in countries]
	results = {}
	for country in countries:
		res = ppr_top10(G, country)
		results[country] = res
  
	with open("ppr_countries.jsonl", "w") as f:
		for country, res in results.items():
			f.write(json.dumps({"country": country, "top10": res}) + "\n")
	return results

def centrality_plots(G, betweenness_score, num=25, directed=True):
	"""
	Generate centrality plots for the graph.
	
	Args:
		G (nx.DiGraph): The graph to analyze.
		num (int): The number of top nodes to display in the plots.
		directed (bool): Whether the graph is directed.
	
	Returns:
		dict: A dictionary containing the centrality data for plotting.
	"""
	top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:num]
	nodes, degrees = zip(*top_nodes)
	top_degrees = [d for n, d in top_nodes]
	xlabel = "Degree"
	ylabel = "Node"
	title = f"Top {num} Nodes by Degree"
	filename = f"top_{num}_nodes_by_degree.png"
	barplot(nodes, top_degrees, title, xlabel, ylabel, filename)
 
	bc_series = pd.Series(betweenness_score)
	top_betweenness = bc_series.sort_values(ascending=False).head(num)
	nodes_bc, betweenness_values = zip(*top_betweenness.items())
	xlabel_bc = "Betweenness Centrality"
	ylabel_bc = "Node"
	title_bc = f"Top {num} Nodes by Betweenness Centrality"
	filename_bc = f"top_{num}_nodes_by_betweenness_centrality.png"
	barplot(nodes_bc, betweenness_values, title_bc, xlabel_bc, ylabel_bc, filename_bc, color='orange')
	if directed:
		in_degree = G.in_degree()
		out_degree = G.out_degree()
		top_in_degree = sorted(in_degree, key=lambda x: x[1], reverse=True)[:num]
		top_out_degree = sorted(out_degree, key=lambda x: x[1], reverse=True)[:num]
		
		nodes_in, in_degrees = zip(*top_in_degree)
		nodes_out, out_degrees = zip(*top_out_degree)
		
		xlabel_in = "In-Degree"
		ylabel_in = "Node"
		title_in = f"Top {num} Nodes by In-Degree"
		filename_in = f"top_{num}_nodes_by_in_degree.png"
		barplot(nodes_in, in_degrees, title_in, xlabel_in, ylabel_in, filename_in)
		
		xlabel_out = "Out-Degree"
		ylabel_out = "Node"
		title_out = f"Top {num} Nodes by Out-Degree"
		filename_out = f"top_{num}_nodes_by_out_degree.png"
		barplot(nodes_out, out_degrees, title_out, xlabel_out, ylabel_out, filename_out)
	
def barplot(nodes, values, title, xlabel, ylabel, filename, color ='skyblue'):
	"""
	Create a bar plot for the given nodes and values.
	
	Args:
		nodes (list): The list of nodes.
		values (list): The list of values corresponding to the nodes.
		title (str): The title of the plot.
		xlabel (str): The label for the x-axis.
		ylabel (str): The label for the y-axis.
		filename (str): The filename to save the plot.
	"""
	plt.figure(figsize=(14, 8), dpi=300)  # Increased size and resolution
	plt.barh(nodes, values, color=color)
	plt.xlabel(xlabel, fontsize=18)
	plt.ylabel(ylabel, fontsize=18)
	plt.title(title, fontsize=20)
	plt.xticks(fontsize=16)  # Increase x-axis tick label size
	plt.yticks(fontsize=14)
	plt.gca().invert_yaxis()  # Highest degree at the top

	# Save with high resolution
	plt.tight_layout()
	plt.savefig(f"output/{filename}", dpi=300)
	plt.show()
 
def betweenness_centrality(G):
	os.environ["NX_CUGRAPH_AUTOCONFIG"] = "1"
	print("Loading GraphML...")
	nx_graph = nx.read_graphml("final_mapped_cleaned_graph.graphml")

	print("Converting to cuGraph...")
	G = nxcg.from_networkx(nx_graph)

	print("Computing Betweenness Centrality...")
	betweenness_centrality = nxcg.betweenness_centrality(G, normalized=True)
	return betweenness_centrality

def compute_community_stats(nodes, G, N_nodes=5):
	subgraph = G.subgraph(nodes)
	degrees = dict(subgraph.degree())
	avg_degree = np.mean(list(degrees.values()))
	avg_clustering = nx.average_clustering(subgraph)
	betweenness = nx.betweenness_centrality(subgraph, k=30, seed=42)  # Approximate
	top_betweenness = sorted(betweenness.values(), reverse=True)[:N_nodes]
	avg_betweenness = np.mean(top_betweenness) if top_betweenness else 0
	return avg_degree, avg_clustering, avg_betweenness

def describe_communities_with_plots_complex (G, N=10, N_nodes=5):
	"""
	Detect and describe the top N communities in graph G based on key nodes, with integrated plots.
	Adds separate plots for average node degree, average clustering coefficient, and betweenness centrality over all communities.
	
	Args:
	- G (networkx.Graph): The graph to analyze.
	- N (int): The number of top communities to describe and plot.
	- N_nodes (int): The number of top nodes to highlight per community.
	- data_dir (str): Directory to save the plots.
	"""
	# Detect communities using the Louvain method
	if type(G) == nx.DiGraph:
		G = nx.to_undirected(G)
	
	# Convert to cuGraph-compatible NetworkX object
	edges = nx.to_pandas_edgelist(G)  # columns: source, target
	gdf_edges = cudf.DataFrame.from_pandas(edges)
	print(f"Number of edges in the graph: {len(gdf_edges)}")
	# Step 3: Create a cuGraph Graph and load edge list
	G_cu = cugraph.Graph()
	G_cu.from_cudf_edgelist(gdf_edges, source='source', destination='target')
	# Ensure the graph is undirected
	# Step 4: Run Leiden community detection
	print("Running Leiden community detection...")
	leiden_df, modularity = cugraph.leiden(G_cu)

	# Step 5: Convert to dict
	partition = dict(zip(leiden_df['vertex'].to_pandas(), leiden_df['partition'].to_pandas()))
	# Invert the partition to get nodes per community
	communities = {}
	print(f"Number of communities detected: {len(set(partition.values()))}")
	for node, comm_id in partition.items():
		communities.setdefault(comm_id, []).append(node)

	# Sort communities by size and get all sizes
	all_communities_sorted = sorted(communities.values(), key=len, reverse=True)
	all_sizes = [len(c) for c in all_communities_sorted]

	# Arrays to hold statistics for all communities
	results = Parallel(n_jobs=-1)(
	delayed(compute_community_stats)(nodes, G, N_nodes)
	for nodes in tqdm(all_communities_sorted, desc="Parallel stats")
	)

	avg_degrees, avg_clusterings, top_betweenness_values = zip(*results)

	# Create integrated plot with subplots
	fig, axs = plt.subplots(2, 2, figsize=(15, 8))  # Adjust for a 2x2 subplot layout

	# Plot size of all communities
	axs[0, 0].bar(range(len(all_sizes)), all_sizes, color='skyblue')
	axs[0, 0].set_title('Size of All Communities')
	axs[0, 0].set_xlabel('Community Index')
	axs[0, 0].set_ylabel('Size (Number of Nodes)')

	# Plot average node degree for each community
	axs[0, 1].bar(range(len(avg_degrees)), avg_degrees, color='lightgreen')
	axs[0, 1].set_title('Average Node Degree for Each Community')
	axs[0, 1].set_xlabel('Community Index')
	axs[0, 1].set_ylabel('Average Degree')

	# Plot average clustering coefficient for each community
	axs[1, 0].bar(range(len(avg_clusterings)), avg_clusterings, color='lightblue')
	axs[1, 0].set_title('Average Clustering Coefficient for Each Community')
	axs[1, 0].set_xlabel('Community Index')
	axs[1, 0].set_ylabel('Average Clustering Coefficient')

	# Plot average betweenness centrality for top nodes in each community
	axs[1, 1].bar(range(len(top_betweenness_values)), top_betweenness_values, color='salmon')
	axs[1, 1].set_title('Average Betweenness Centrality for Top Nodes in Each Community')
	axs[1, 1].set_xlabel('Community Index')
	axs[1, 1].set_ylabel('Average Betweenness Centrality')

	plt.tight_layout()
	plt.savefig(f'output/community_statistics_overview.svg')
	plt.show()

	# Determine subplot grid size
	rows = math.ceil(N / 2)
	cols = 2 if N > 1 else 1  # Use 2 columns if N > 1, else just 1

	# Create integrated plot with subplots for top N communities
	fig, axes = plt.subplots(rows, cols, figsize=(10 * cols,12 * rows), squeeze=False)
	for i, nodes in enumerate(all_communities_sorted[:N], start=0):
		subgraph = G.subgraph(nodes)
		degrees = dict(subgraph.degree())
		sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
		key_nodes, key_degrees = zip(*sorted_nodes[:N_nodes])  # Adjust number as needed

		# Select the appropriate subplot
		ax = axes[i // cols, i % cols]
		bars = ax.bar(range(len(key_nodes)), key_degrees, tick_label=key_nodes, color='lightgreen')
		ax.set_title(f'Community {i+1} (Top Nodes by Degree)', fontsize=18)
		ax.set_xlabel('Node label', fontsize=18)
		ax.set_ylabel('Degree', fontsize=18)
		ax.tick_params(axis='x', labelsize=18, rotation=45)
		ax.tick_params(axis='y', labelsize=18)

	plt.tight_layout()
	plt.savefig(f'output/top_nodes_by_degree_combined.svg')
	plt.show()
 
 
def plot_log_binned_degree_distribution(G, logbase=2):
	degrees = [d for n, d in G.degree()]
	fit = powerlaw.Fit(degrees, discrete=True, fit_method='Likelihood')
	print(f"Alpha (γ): {fit.power_law.alpha}")
	print(f"xmin: {fit.power_law.xmin}")
	R, p = fit.distribution_compare('power_law', 'lognormal')
	print(f"Loglikelihood ratio: {R}, p-value: {p}")

	max_deg = max(degrees)
	
	# Create logarithmic bins
	bins = np.logspace(0, np.log10(max_deg), num=50, base=logbase)
	hist, bin_edges = np.histogram(degrees, bins=bins, density=True)

	# Use midpoints for plotting
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

	plt.figure(figsize=(8, 6), dpi=300)
	plt.loglog(bin_centers, hist, marker='o', linestyle='None')
	plt.xlabel("Degree k (log scale)", fontsize=16)
	plt.ylabel("P(k) (log scale)", fontsize=16)
	plt.title("Log-binned Degree Distribution", fontsize=18)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("output/powerlaw_fit.png", dpi=300)
	plt.show()