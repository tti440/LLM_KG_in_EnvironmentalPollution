from LLM_Graph.extraction_process import fulltext_2_chunks
import argparse


if __name__ == "__main__":
	argparse = argparse.ArgumentParser(description="Run the preprocess from raw text into chunks.")
	argparse.add_argument("--files", help="Takes a list of files to process, separated by commas without any spaces.", type=str, required=True)
	argparse.add_argument("--output_file", help="The output file to save the chunks. It takes .jsonl format", type=str, default=None)
	argparse.add_argument("--chunk_size", help="The maximum size of each chunk in words.", type=int, default=750)
	args = argparse.parse_args()
	files = args.files.split(",")
	if not files:
		raise ValueError("No files provided. Please provide a list of files to process.")
	if args.output_file is not None:
		assert ".jsonl" in args.output_file, "Output file must be in .jsonl format."
	fulltext_2_chunks(files, args.output_file, args.chunk_size)