from careerpathway.scoring.diversity import *
from careerpathway.scoring.similarity import Similarity
from careerpathway.scoring.load_testset import *
from careerpathway.scoring.utils import *

__all__ = ['Diversity', 'Similarity']


FILENAMES = [
    # 'meta-llama_Meta-Llama-3-70B-Instruct.jsonl'
    #    'meta-llama_Llama-2-13b-chat-hf.jsonl',
    #'mistralai_Mistral-7B-Instruct-v0.1.jsonl',
    #'mistralai_Mistral-7B-Instruct-v0.2.jsonl',
    #'meta-llama_Llama-2-7b-chat-hf.jsonl',
    #'meta-llama_Meta-Llama-3-8B-Instruct.jsonl',
    #'mistralai_Mistral-7B-Instruct-v0.3.jsonl',
    'gpt-4o-mini.jsonl',
    'google_gemma-2-2b-it.jsonl',
    'google_gemma-2-9b-it.jsonl',
    'google_gemma-2-27b-it.jsonl',
    'Qwen_Qwen2.5-0.5B-Instruct.jsonl',
    'Qwen_Qwen2.5-1.5B-Instruct.jsonl',
    'Qwen_Qwen2.5-3B-Instruct.jsonl',
    'Qwen_Qwen2.5-7B-Instruct.jsonl',
    'Qwen_Qwen2.5-14B-Instruct.jsonl',
    'Qwen_Qwen2.5-32B-Instruct.jsonl',
    'CohereForAI_aya-expanse-8b.jsonl',
    'CohereForAI_aya-expanse-32b.jsonl',
    'Qwen_Qwen2.5-72B-Instruct.jsonl',
]
