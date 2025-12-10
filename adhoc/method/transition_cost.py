import numpy as np
from mylmeval.utils import open_json
import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict
from careerpathway.scoring.load_testset import load_high_qual_diversity
from typing import List, Tuple, Dict, Set, Any, Optional, Union, DefaultDict, Iterator
import re

class TransitionCostModel:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', 
                similarity_threshold: float = 0.9) -> None:
        # 임베딩 모델 로드
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.similarity_threshold = similarity_threshold
        
        # 그래프 및 임베딩 저장소 초기화
        self.embeddings_cache = {}  # 메모리에 적재된 임베딩의 캐시
        
        # 전환 카운트 저장소
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.source_total_counts = defaultdict(int)  # 소스 노드별 총 전환 수
        
        # 노드 클러스터링을 위한 저장소
        self.node_clusters = {}  # 노드 -> 클러스터 ID
        self.cluster_members = defaultdict(list)  # 클러스터 ID -> 멤버 노드 리스트
    

    def _calculate_transition_cost_text(self, source: str, target: str) -> float:

        # 소스와 타겟에 대한 요구사항 가져오기
        source_reqs = set(self.requirement_data.get(source, []))
        target_reqs = set(self.requirement_data.get(target, []))
        
        # 요구사항이 없는 경우 최대 비용 반환
        if not source_reqs or not target_reqs:
            return 1.0
        
        # 타겟에만 있는 요구사항 계산 (차집합)
        additional_reqs = target_reqs - source_reqs
        
        # 전환 비용 계산: 추가 요구사항 / 타겟 총 요구사항
        if not target_reqs:
            return 0.0
        
        return len(additional_reqs) / len(target_reqs)

    def _calculate_transition_cost_emb(self, source: str, target: str) -> float:

        # 소스와 타겟 임베딩 확인 또는 생성
        if source not in self.embeddings_cache:
            source_text = " ".join(self.requirement_data.get(source, []))
            if not source_text:
                return 1.0
            self.embeddings_cache[source] = self.embedding_model.encode(source_text)
            
        if target not in self.embeddings_cache:
            target_text = " ".join(self.requirement_data.get(target, []))
            if not target_text:
                return 1.0
            self.embeddings_cache[target] = self.embedding_model.encode(target_text)
        
        # 임베딩 간 코사인 유사도 계산
        similarity = cosine_similarity(
            [self.embeddings_cache[source]], 
            [self.embeddings_cache[target]]
        )[0][0]
        
        # 유사도를 거리(비용)로 변환 (1 - 유사도)
        return 1.0 - similarity

    def upload_requirement_data(self, data: List[Dict[str, str]]) -> None:

        # 요구사항 데이터 저장
        self.requirement_data = data
        
        # 임베딩 캐시 초기화 (필요시 여기서 사전 계산도 가능)
        self.embeddings_cache = {}
        
        # 효율성을 위해 일괄 임베딩 계산 (선택적)
        print("요구사항 임베딩 계산 중...")
        batch_size = 50
        all_items = [(r['job'], r['skills']) for r in data]
        
        for batch in tqdm.tqdm(batch_generator(all_items, batch_size)):
            for name, requirements in batch:
                if not requirements:
                    continue
                    
                # 요구사항을 하나의 텍스트로 결합
                combined_text = " ".join(requirements)
                
                # 임베딩 계산 및 캐싱
                self.embeddings_cache[name] = self.embedding_model.encode(combined_text)
        
        print(f"총 {len(self.embeddings_cache)} 개의 임베딩이 계산되었습니다.")
        
    def calculate_transition_cost(self, source: str, target: str) -> Tuple[float, float]:
        emb_cost = self._calculate_transition_cost_emb(source, target)
        text_cost = self._calculate_transition_cost_text(source, target)
        return text_cost, emb_cost
                
    def _find_similar_nodes(self, node_name: str, k: int = 5) -> List[Tuple[str, float]]:

        if node_name not in self.embeddings_cache:
            return []
            
        node_embedding = self.embeddings_cache[node_name]
        
        # 모든 노드 탐색 (노드 타입 구분 없이)
        all_nodes = [n for n in self.embeddings_cache.keys() if n != node_name]
        
        if not all_nodes:
            return []
        
        # 메모리 효율적인 방식으로 유사도 계산
        similarities = []
        
        # 배치 처리로 메모리 사용량 최적화
        batch_size = 50
        for i in range(0, len(all_nodes), batch_size):
            batch = all_nodes[i:i+batch_size]
            
            # 배치 임베딩 로드
            batch_embeddings = []
            for other_name in batch:
                try:
                    emb = self.embeddings_cache[other_name]
                    batch_embeddings.append((other_name, emb))
                except:
                    # 임베딩 로드 실패시 건너뛰기
                    continue
            
            # 유사도 계산
            for other_name, other_embedding in batch_embeddings:
                similarity = cosine_similarity(
                    [node_embedding], 
                    [other_embedding]
                )[0][0]
                    
                if similarity >= self.similarity_threshold:
                    similarities.append((other_name, similarity))
        
        # 유사도 내림차순으로 정렬하고 상위 k개 반환
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]


def batch_generator(data, batch_size=100):
    """데이터를 배치로 나누는 제너레이터"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + min(batch_size, len(data) - i)]

# 사용 예시 함수
def main(
    similarity_threshold: float = 0.90,
    requirement_data_path: str = '/Users/sb/Downloads/data13_15_kaggle.jsonl'
    ) -> None:
    """
    메모리 효율적인 방식의 모델 사용 예시 데모 함수
    """
    
    model = TransitionCostModel(similarity_threshold=similarity_threshold)
    
    def run(testset: Dict[str, Any]) -> None:
        keyword = testset['keyword']
        queries = [(keyword, target) for target in testset['targets']]
        
        data = open_json(requirement_data_path)
        data = [{'job' : r['job'], 'skills' : r['skills']} for r in data]
        model.upload_requirement_data(data)
        
        for source, target in queries:
            cost_text, cost_emb = model.calculate_transition_cost(source, target)
            print(f"{source} → {target}: text: {cost_text:.4f}, emb: {cost_emb:.4f}")

    keywords = [
        'Computer Science',
        'Psychology',
        'Medicine',
        'Engineering',
        'Philosophy',
        'Economics',
        'Biology',
    ]
    jobs = [
        'Data Scientist', 'Software Engineer', 'Psychologist', 'Doctor', 'Professor', 'Economist', 'Biologist',
        'Data Analyst', 'Web Developer', 'Therapist', 'Nurse', 'Writer', 'Accountant', 'Researcher',
        'Machine Learning Engineer', 'Database Administrator', 'Counselor', 'Pharmacist', 'Artist', 'Financial Analyst', 'Scientist',
        'Network Administrator', 'Social Worker', 'Dentist', 'Journalist', 'Investment Banker', 'Chemist'
    ]
    
    testset = [{'keyword': keyword, 'targets': jobs} for keyword in keywords]
    for test in testset:
        run(test)
        

if __name__ == "__main__":
    import fire
    fire.Fire(main)