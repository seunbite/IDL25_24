import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Any, Optional
from tqdm import tqdm




def calculate_diversity_metrics(embeddings: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    def compute_pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
        # 정규화된 코사인 거리 계산
        normalized_embeddings = normalize(embeddings)
        distances = squareform(pdist(normalized_embeddings, metric='cosine'))
        return distances
    
    def compute_distance_entropy(distances: np.ndarray) -> float:
        # 거리 분포의 엔트로피 계산
        hist, _ = np.histogram(distances.flatten(), bins=20, density=True)
        hist = hist[hist > 0]  # 0이 아닌 값만 사용
        return float(entropy(hist))
    
    distances = compute_pairwise_distances(embeddings)
    return {
        'entropy': compute_distance_entropy(distances),
        'mean_distance': float(np.mean(distances)),
        'variance': float(np.var(distances)),
        'distances': distances
    }


class Diversity:
    def __init__(self, embedding_model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(embedding_model_name)
    
    def _batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    
    def _2d_embedding(self, prediction_sets: List[List[str]], batch_size: int = 32) -> List[np.ndarray]:
        embeddings = []
        for predictions in tqdm(prediction_sets, desc="Encoding predictions"):
            if len(predictions) > 0:
                embeddings.append(self._batch_encode(predictions, batch_size))
        return embeddings
    
    def evaluate(self, 
                prediction_sets: List[List[str]], 
                batch_size: int = 32,
                return_all: bool = False) -> Dict[str, Union[float, List[float]]]:

        embeddings = self._2d_embedding(prediction_sets, batch_size)
        results: Dict[str, List[Any]] = {'entropy': [], 'mean_distance': [], 
                                        'variance': [], 'distances': []}
        
        # 배치 처리로 메트릭 계산
        for embedding in tqdm(embeddings, desc="Computing metrics"):
            metrics = calculate_diversity_metrics(embedding)
            for key in results.keys():
                results[key].append(metrics[key])
        
        # 최종 결과 계산
        final_results = {}
        for key in ['entropy', 'mean_distance', 'variance']:
            values = np.array(results[key])
            final_results[key] = float(np.mean(values))
            print(f'{key}: {final_results[key]}')
        
        if return_all:
            return {**results, 'input_sets' : prediction_sets}

        return final_results

    def the_most_diverse(self,
                        prediction_sets: List[List[str]],
                        n: int = 5,
                        batch_size: int = 32) -> Dict[str, List[int]]:
        
        embeddings = self._2d_embedding(prediction_sets, batch_size)
        
        results: Dict[str, List[Any]] = {'entropy': [], 'mean_distance': [], 
                                        'variance': [], 'distances': []}

        for embedding in tqdm(embeddings, desc="Computing metrics"):
            metrics = calculate_diversity_metrics(embedding)
            for key in results.keys():
                results[key].append(metrics[key])
                
        # 가장 다양한 예측 결과 선택
        for metric in ['entropy', 'mean_distance', 'variance']:
            values = np.array(results[metric])
            highest_indices = np.argsort(values)[:n]
            lowest_indices = np.argsort(values)[::-1][:n]
            for i in highest_indices:
                print('-------------')
                print('Highest', metric, i, values[i])
                print('-------------')
                print('\n'.join(prediction_sets[i]))
            for i in lowest_indices:
                print('-------------')
                print('Lowest', metric, i, values[i])
                print('-------------')
                print('\n'.join(prediction_sets[i]))
    
        return results