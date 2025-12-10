from typing import List, Dict, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from rouge_score.rouge_scorer import RougeScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from nltk.translate.meteor_score import meteor_score
import torch
from dataclasses import dataclass
import warnings
from transformers import logging
from tqdm import trange

logging.set_verbosity_error()
warnings.filterwarnings('ignore')

try:
    nltk.download('punkt')
    nltk.download('wordnet')  # METEOR 스코어 계산에 필요
except:
    pass


@dataclass
class MetricScores:
    bleu: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    meteor: float
    bert_score: float


class TextDataset(Dataset):
    def __init__(self, predictions: List[List[str]], references: List[str]):
        assert len(predictions) == len(references)
        self.predictions = predictions  # List[List[str]]
        self.references = references    # List[str]
        
    def __len__(self):
        return len(self.references)
    
    def __getitem__(self, idx):
        return self.predictions[idx], self.references[idx]  # Returns (List[str], str)




class Similarity:
    def __init__(self, batch_size: int = 128, use_gpu: bool = True):
        self.batch_size = batch_size
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = sentence_bleu

    def compute_bert_score(self, predictions: List[str], references: List[str], lang: str = 'en') -> np.ndarray:
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        if len(valid_pairs)==0:
            return np.array([])
        else:
            valid_predictions, valid_references = zip(*valid_pairs)
            P, R, F1 = score(valid_predictions, valid_references, lang=lang, batch_size=self.batch_size, device=self.device)
            return F1.numpy()

    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, List[float]]:
        results = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(pred, ref)
            results['rouge1'].append(scores['rouge1'].fmeasure)
            results['rouge2'].append(scores['rouge2'].fmeasure)
            results['rougeL'].append(scores['rougeL'].fmeasure)
        return results

    def compute_meteor_score(self, predictions: List[str], references: List[str]) -> List[float]:
        scores = []
        for pred, ref in zip(predictions, references):
            score = meteor_score([ref.split()], pred.split())
            scores.append(score)
        return scores

    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> List[float]:
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.bleu([ref.split()], pred.split())
            scores.append(score)
        return scores

    def evaluate(self, predictions: List[str], references: List[str], return_all: bool = False, lang: str = 'en') -> Dict:
        assert len(predictions) == len(references), "Predictions and references must have same length"
        print('evaluation started!')
        print(predictions[0])
        print(references[0])
        
        # 배치 크기로 데이터 분할
        results = {}
        for i in trange(0, len(predictions), self.batch_size):
            batch_pred = predictions[i:i + self.batch_size]
            batch_ref = references[i:i + self.batch_size]
            
            # BLEU 점수 계산
            batch_bleu = self.compute_bleu_score(batch_pred, batch_ref)
            results.setdefault('bleu', []).extend(batch_bleu)
            
            # ROUGE 점수 계산
            batch_rouge = self.compute_rouge_scores(batch_pred, batch_ref)
            results.setdefault('rouge_1', []).extend(batch_rouge['rouge1'])
            results.setdefault('rouge_2', []).extend(batch_rouge['rouge2'])
            results.setdefault('rouge_l', []).extend(batch_rouge['rougeL'])
            
            # METEOR 점수 계산
            batch_meteor = self.compute_meteor_score(batch_pred, batch_ref)
            results.setdefault('meteor', []).extend(batch_meteor)
            
            # BERTScore 계산
            batch_bert = self.compute_bert_score(batch_pred, batch_ref, lang=lang)
            results.setdefault('bert_score', []).extend(batch_bert)

        if return_all:
            return results
        else:
            return {key: np.mean(value) for key, value in results.items()}

    
    def group_evaluate(self, predictions: List[List[str]], references: List[str], the_most_similar: int = 5) -> List[MetricScores]:
        assert len(predictions) == len(references), "Predictions and references must have same length"
        results = {key: [] for key in MetricScores.__annotations__.keys()}
        print('fist example of group evaluation, predictions (list) and references (str)')
        print(predictions[0])
        print(references[0])
        
        # 각 그룹의 시작 인덱스 계산
        start_indices = [0]
        for group in predictions[:-1]:
            start_indices.append(start_indices[-1] + len(group))
        
        # 모든 예측에 대한 점수 계산
        flattened_predictions = [p for group in predictions for p in group]
        flattened_references = []
        for i, ref in enumerate(references):
            flattened_references.extend([ref] * len(predictions[i]))
        
        all_scores = self.evaluate(flattened_predictions, flattened_references, return_all=True)
        
        # 그룹별로 처리
        for i in range(len(references)):
            start_idx = start_indices[i]
            end_idx = start_indices[i] + len(predictions[i])
            
            # 현재 그룹의 BERTScore
            group_bert_scores = all_scores['bert_score'][start_idx:end_idx]
            
            # BERTScore 기준으로 상위 k개 인덱스 찾기
            k = min(the_most_similar, len(group_bert_scores))  # 그룹 크기가 k보다 작을 경우 처리
            top_k_indices = np.argsort(group_bert_scores)[-k:]
            
            # 각 메트릭에 대해 상위 k개 결과만 저장
            for key in results.keys():
                group_scores = all_scores[key][start_idx:end_idx]
                results[key].extend([group_scores[idx] for idx in top_k_indices])
        
        # 최종 평균 계산
        return {key: np.mean(value) for key, value in results.items()}