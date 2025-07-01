import pandas as pd
import json
import os



def data_preprocess():
	path = "Industrial_Reporting_Facilities_relevant_contribution.xlsx"
	df = pd.read_excel(path, sheet_name="Facilities_relevant_contributio")
	erptr_path = "erptr_industry_code.json"
	erptr = json.load(open(erptr_path, "r", encoding="utf-8"))
	column = df["EPRTR_AnnexI_MainActivity"]
	indistry_names=[]
	for entry in column:
		index = entry.find(")")
		industry_name = entry[:index+1].strip()  # Extract the part before the first parenthesis
		indistry_names.append(erptr[industry_name]["main_activity"])
	df["EPRTR_AnnexI_MainActivity"]= indistry_names
	df.to_csv("Updated_Industrial_Reporting_Facilities_relevant_contribution.xlsx", index=False, encoding="utf-8")
 
def get_term_mapping(df: pd.DataFrame):
	terms = set()
	for term in df["countryName"]:
		terms.add(term.lower())
	for term in df["Pollutant"]:
		terms.add(term.lower())
	for term in df["EPRTR_AnnexI_MainActivity"]:
		terms.add(term.lower())
	for term in df["Sector"]:
		terms.add(term.lower())