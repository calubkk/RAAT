import re
import ftfy
from collections import Counter
from typing import List, Dict
import string

def normalize_answer(s):
    def remove_(text):
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub("《", " ", text)
        text = re.sub("》", " ", text)
        text = re.sub("<", " ", text)
        text = re.sub(">", " ", text)
        text = re.sub("〈", " ", text)
        text = re.sub("〉", " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """计算F1分数。"""
    prediction_tokens = normalize_answer(prediction).split()
    #print(prediction_tokens)
    ground_truth_tokens = normalize_answer(ground_truth).split()
    #print(ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    #print(common)
    num_same = sum(common.values())
    #print(num_same)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    #print(precision)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    #print(recall)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction: str, ground_truths: List[str]) -> bool:
    """计算精确匹配得分。"""
    return normalize_answer(prediction) in [normalize_answer(gt) for gt in ground_truths]

def compute_metrics(predicted_answer: str, ground_truth_answers: List[str]) -> Dict:
    """计算并返回EM和F1分数。"""
    em_score = exact_match_score(predicted_answer, ground_truth_answers)
    f1_scores = [f1_score(predicted_answer, gt) for gt in ground_truth_answers]
    max_f1 = max(f1_scores) if f1_scores else 0.0
    #print({"em": 1.0 if em_score else 0.0, "f1": max_f1})
    return {"em": 1.0 if em_score else 0.0, "f1": max_f1}

'''
predicted = "some answer"
ground_truths = ["correct answer 1", "correct answer 2"]
metrics = compute_metrics(predicted, ground_truths)
print(metrics)
'''