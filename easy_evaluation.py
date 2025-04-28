import json
from json import JSONDecodeError
import csv
from typing import List

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def load_predictions(pred_file: str) -> List[str]:
    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f if line.strip()]
    return predictions


def load_references(ref_file: str) -> List[str]:
    """Loads references from a JSONL file.

    Reads each line from the specified file, parses it as a JSON object,
    and extracts the 'lay_summary' field.

    Args:
        ref_file (str): Path to the JSONL file containing references.

    Returns:
        List[str]: A list of reference summaries.

    Raises:
        ValueError: If the JSON data is not a list or dict, or if 'lay_summary' key is missing.
        JSONDecodeError: If any line in the file is not valid JSON.
    """
    references = []
    with open(ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise JSONDecodeError(f"Invalid JSON on line: {line.strip()}") from e

            if isinstance(data, dict):
                if 'summary' in data:
                    references.append(data['summary'])
                else:
                    raise ValueError("Reference JSON must contain 'summary' key.")
            else:
                raise ValueError("Each line in the JSONL file must contain a valid JSON object.")

    return references


def compute_rouge(predictions: List[str], references: List[str]):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    detailed_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(pred, ref)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougeL_scores.append(rougeL)
        detailed_scores.append((rouge1, rouge2, rougeL))

    rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return rouge1, rouge2, rougeL, detailed_scores


def compute_bleu(predictions: List[str], references: List[str]):
    smoothie = SmoothingFunction().method4
    list_of_references = [[ref.split()] for ref in references]
    hypotheses = [pred.split() for pred in predictions]

    bleu_score = corpus_bleu(list_of_references, hypotheses, smoothing_function=smoothie)
    return bleu_score


def compute_individual_bleu(predictions: List[str], references: List[str]):
    smoothie = SmoothingFunction().method4
    individual_bleu_scores = []

    for pred, ref in zip(predictions, references):
        bleu = corpus_bleu([[ref.split()]], [pred.split()], smoothing_function=smoothie)
        individual_bleu_scores.append(bleu)

    return individual_bleu_scores


def save_scores(predictions: List[str], references: List[str], detailed_rouge: List[tuple], detailed_bleu: List[float], output_file: str):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Prediction', 'Reference', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU'])
        for pred, ref, (r1, r2, rL), bleu in zip(predictions, references, detailed_rouge, detailed_bleu):
            writer.writerow([pred, ref, f"{r1:.4f}", f"{r2:.4f}", f"{rL:.4f}", f"{bleu:.4f}"])


def main(pred_file: str, ref_file: str):
    predictions = load_predictions(pred_file)
    references = load_references(ref_file)

    # print number of lines in pred_file
    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f if line.strip()]
    print(f"Number of lines in {pred_file}: {len(predictions)}")
    # print number of lines in ref_file
    with open(ref_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f if line.strip()]
    print(f"Number of lines in {ref_file}: {len(references)}")
    
    assert len(predictions) == len(references), "Predictions and references must have the same length."

    rouge1, rouge2, rougeL, detailed_rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    individual_bleu = compute_individual_bleu(predictions, references)

    print(f"ROUGE-1: {rouge1:.4f}")
    print(f"ROUGE-2: {rouge2:.4f}")
    print(f"ROUGE-L: {rougeL:.4f}")
    print(f"BLEU: {bleu:.4f}")

    save_scores(predictions, references, detailed_rouge, individual_bleu, "detailed_scores.csv")
    print("Detailed scores saved to detailed_scores.csv")


if __name__ == "__main__":


    pred_file = "elife.txt"
    ref_file = "elife_train.json"
    main(pred_file, ref_file)
