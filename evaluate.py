import os, sys, json, logging
import textstat
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
import nltk 
from alignscore import AlignScore
from lens.lens_score import LENS
import torch
from summac.model_summac import SummaCConv

from lens import download_model, LENS


nltk.download('punkt')

def calc_rouge(preds, refs):
	
	# Get ROUGE F1 scores
	scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], \
									use_stemmer=True, split_summaries=True)
	# Only score predictions that have corresponding references
	num_samples = min(len(preds), len(refs))
	scores = [scorer.score(preds[i], refs[i]) for i in range(num_samples)]
	return np.mean([s['rouge1'].fmeasure for s in scores]), \
		   np.mean([s['rouge2'].fmeasure for s in scores]), \
		   np.mean([s['rougeLsum'].fmeasure for s in scores])

def calc_bertscore(preds, refs, device=None):
	"""Calculate BERTScore for a set of predictions and references.

	This function computes the BERTScore F1 score, which measures the similarity
	between predicted and reference summaries using contextualized embeddings.

	Args:
		preds (list of str): List of predicted summaries.
		refs (list of str): List of reference summaries.
		device (str, optional): Device to use for computation (e.g., 'cuda:0', 'cpu').
			Defaults to GPU if available, otherwise CPU.

	Returns:
		float: The mean BERTScore F1 score.
	"""
	logging.info("Calculating BERTScore...")
	if device is None:
		device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	
	min_len = min(len(preds), len(refs))
	preds = preds[:min_len]
	refs = refs[:min_len]
	
	P, R, F1 = score(preds, refs, lang="en", verbose=True, device=device)
	return np.mean(F1.tolist())

def calc_readability(preds):
	
	fkgl_scores = []
	cli_scores = []
	dcrs_scores = []
	for pred in preds:
		fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
		cli_scores.append(textstat.coleman_liau_index(pred))
		dcrs_scores.append(textstat.dale_chall_readability_score(pred))
	return np.mean(fkgl_scores), np.mean(cli_scores), np.mean(dcrs_scores)

def calc_lens(preds, refs, docs, device=None):
	
	if device is None:
		device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

	model_path = os.path.join(os.path.dirname(__file__), "models/LENS/checkpoints/epoch=5-step=6102.ckpt")

	model_path = download_model("davidheineman/lens")

	metric = LENS(model_path, rescale=True, device=device)
	abstracts = [d.split("\n")[0] for d in docs]
	refs = [[x] for x in refs]

	# scores = metric.score(abstracts, preds, refs, batch_size=8, gpus=1)
	scores = metric.score(abstracts, preds, refs, batch_size=8)
	return np.mean(scores)

def calc_alignscore(preds, docs):
	
	alignscorer = AlignScore(model='roberta-base', batch_size=16, device='cuda:0', \
                           ckpt_path='./models/AlignScore/AlignScore-base.ckpt', evaluation_mode='nli_sp')
	return np.mean(alignscorer.score(contexts=docs, claims=preds))

def cal_summac(preds, docs):
	
	model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
	return np.mean(model_conv.score(docs, preds)['scores'])

def read_file_lines(path):
	logging.info(f"Reading file: {path}")
	with open(path, 'r', encoding='utf-8') as f:
		lines = f.readlines()

	if path.endswith('.jsonl'):
		lines = [json.loads(line) for line in lines]

	return lines

def evaluate(pred_path, gold_path):
	logging.info("Evaluating predictions...")
	# Load data from files
	refs_dicts = read_file_lines(gold_path)
	preds = read_file_lines(pred_path)

	refs = [d['lay_summary'] for d in refs_dicts]
	docs = [d['article'] for d in refs_dicts]

	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	score_dict = {}

	# Relevance scores
	logging.info("Calculating ROUGE scores...")
	rouge1_score, rouge2_score, rougel_score = calc_rouge(preds, refs)
	score_dict['ROUGE1'] = rouge1_score
	score_dict['ROUGE2'] = rouge2_score
	score_dict['ROUGEL'] = rougel_score
	score_dict['BERTScore'] = calc_bertscore(preds, refs)

	# # Readability scores
	logging.info("Calculating readability scores...")
	fkgl_score, cli_score, dcrs_score = calc_readability(preds)
	score_dict['FKGL'] = fkgl_score
	score_dict['DCRS'] = dcrs_score
	score_dict['CLI'] = cli_score

	# logging.info("Calculating LENS scores...")
	# score_dict['LENS'] = calc_lens(preds, refs, docs, device=device)

	# Factuality scores
	# logging.info("Calculating AlignScore...")
	# score_dict['AlignScore'] = calc_alignscore(preds, docs)
	# logging.info("Calculating SummaC scores...")
	# score_dict['SummaC'] = cal_summac(preds, docs)

	print("Evaluation Results:")
	for metric, value in score_dict.items():
		print(f"  {metric}: {value:.4f}")


	return score_dict

def write_scores(score_dict, output_filepath):
	# Write scores to file
	with open(output_filepath, 'w', encoding='utf-8') as f:
		for key, value in score_dict.items():
			f.write(f"{key}: {value}\n")


def main():

	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

	submit_dir = sys.argv[1] # directory with txt files ("elife.txt" and "plos.txt") with predictions
	truth_dir = sys.argv[2] # directory with jsonl files containing references and articles 

	output_dir = "./"

	# Calculate + write elife scores
	logging.info("Calculating eLife scores...")


	elife_scores = evaluate(
		os.path.join(submit_dir, 'elife.txt'), 
		os.path.join(truth_dir, 'elife_val.jsonl'),
	)
	write_scores(elife_scores, os.path.join(output_dir, 'elife_scores.txt'))

	torch.cuda.empty_cache()

	# Calculate + write PLOS scores
	# plos_scores = evaluate(
	#   os.path.join(submit_dir, 'plos.txt'), 
	#   os.path.join(truth_dir, 'PLOS_val.jsonl'),
	#   )
	# write_scores(plos_scores, os.path.join(output_dir, 'plos_scores.txt'))

	# Calculate + write overall scores
	#avg_scores = {key: np.mean([elife_scores[key], plos_scores[key]]) for key in elife_scores.keys()}
	avg_scores = {key: elife_scores[key] for key in elife_scores.keys()}
	write_scores(avg_scores, os.path.join(output_dir, 'scores.txt'))

if __name__ == "__main__":
	main()