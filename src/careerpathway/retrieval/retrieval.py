import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
from rank_bm25 import BM25Okapi
from typing import List, Tuple
from tqdm import tqdm, trange

class OptimizedRetrievalMethods:
    def __init__(self, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        self.model.to(self.device)
        self.batch_size = batch_size
        self.tfidf_vectorizer = TfidfVectorizer()
        self.doc_embeddings_cache = {}
        
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch, 
                    convert_to_tensor=True,
                    device=self.device
                )
            embeddings.append(batch_embeddings.cpu().numpy())
        return np.vstack(embeddings)

    def precompute_embeddings(self, documents: List[str]):
        doc_key = hash(tuple(documents))
        if doc_key not in self.doc_embeddings_cache:
            self.doc_embeddings_cache[doc_key] = self.batch_encode(documents)
        return self.doc_embeddings_cache[doc_key]
    
    def batch_cosine_similarity(
        self, 
        query_embeddings: np.ndarray, 
        doc_embeddings: np.ndarray,
        batch_size: int = 100
    ) -> np.ndarray:
        similarities = []
        for i in trange(0, len(query_embeddings), batch_size):
            batch_queries = query_embeddings[i:i + batch_size]
            batch_similarities = []
            
            # 문서도 배치로 처리
            for j in range(0, len(doc_embeddings), self.batch_size):
                batch_docs = doc_embeddings[j:j + self.batch_size]
                # 정규화된 내적으로 코사인 유사도 계산
                norm_q = np.linalg.norm(batch_queries, axis=1)[:, np.newaxis]
                norm_d = np.linalg.norm(batch_docs, axis=1)
                sim = np.dot(batch_queries, batch_docs.T) / np.dot(norm_q, norm_d[np.newaxis, :])
                batch_similarities.append(sim)
            
            # 현재 쿼리 배치의 모든 문서에 대한 유사도 합치기
            similarities.append(np.hstack(batch_similarities))
        
        return np.vstack(similarities)
    
    def semantic_search(self, queries: List[str], documents: List[str], top_k: int = 5) -> List[List[Tuple[int, float, str]]]:
        """배치 처리된 시맨틱 검색"""
        # 문서와 쿼리 임베딩 계산
        doc_embeddings = self.precompute_embeddings(documents)
        query_embeddings = self.batch_encode(queries)
        
        # 배치 처리된 코사인 유사도 계산
        similarities = self.batch_cosine_similarity(query_embeddings, doc_embeddings)
        
        # 각 쿼리에 대한 결과 처리
        all_results = []
        for sim in similarities:
            top_indices = np.argpartition(-sim, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-sim[top_indices])]  # 정확한 순서로 정렬
            results = [(idx, float(sim[idx]), documents[idx]) for idx in top_indices]
            all_results.append(results)
            
        return all_results
    
    def lexical_search(self, queries: List[str], documents: List[str], top_k: int = 5) -> List[List[Tuple[int, float, str]]]:
        """배치 처리된 BM25 검색"""
        tokenized_docs = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        all_results = []
        for query in queries:
            scores = bm25.get_scores(query.split())
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [(idx, scores[idx], documents[idx]) for idx in top_indices]
            all_results.append(results)
            
        return all_results

    def hybrid_search(self, queries: List[str], documents: List[str], top_k: int = 5, 
                     semantic_weight: float = 0.5) -> List[List[Tuple[int, float, str]]]:
        semantic_results = self.semantic_search(queries, documents, top_k=len(documents))
        lexical_results = self.lexical_search(queries, documents, top_k=len(documents))
        
        all_results = []
        for semantic_res, lexical_res in zip(semantic_results, lexical_results):
            semantic_scores = {idx: score for idx, score, _ in semantic_res}
            lexical_scores = {idx: score for idx, score, _ in lexical_res}
            
            combined_scores = {}
            for idx in range(len(documents)):
                semantic_score = semantic_scores.get(idx, 0)
                lexical_score = lexical_scores.get(idx, 0)
                combined_scores[idx] = (semantic_weight * semantic_score + 
                                      (1 - semantic_weight) * lexical_score)
            
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results = [(idx, score, documents[idx]) for idx, score in sorted_results]
            all_results.append(results)
            
        return all_results