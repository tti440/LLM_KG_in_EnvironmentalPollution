from LLM_Graph.pipeline import construction_pipeline
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
	construction_pipeline(args.node_threshold, args.node_embedding_file, args.edge_threshold, args.edge_embedding_file, args.verbose)