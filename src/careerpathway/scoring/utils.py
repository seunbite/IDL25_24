FILENAMES = [
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

def process_str_to_answers(text):
    return [r for r in text.split('\n') if 'Here' not in r and len(r) > 2]

def delete_prefix(text):
    job_prefixes = ['Job 1', 'Job 2', 'Job 3', 'Job 4', 'Job 5', 'Job 6', 'Job 7', 'Job 8', 'Job 9', 'Job 10'] 
    for prefix in job_prefixes:
        text = text.replace(f"{prefix}: ", '')
    if len(text) < 2:
        return ''
    return text
    
    
def nodes_into_string(gt):
    gt = ' '.join([r['content']['main']+ " "+r['content'].get('detail','') for r in gt])
    return gt


def get_gt_from_id(query_graph_idx, diversity_file):
    retrieved = [r for r in diversity_file if r['idx'] == query_graph_idx and r['from'] != None]
    return " ".join(
        [r['content']['main']+ " " + r['content'].get('detail','') for r in retrieved]
        )

def get_prompt_and_model(value):
    if value == 'salary':
        prompt = "Answer the average annual salary for the following job.\n{}"
        model_name_or_path = '/scratch2/snail0822/career/job-salary-qwen-3b'
    elif value == 'fitness':
        prompt = "Answer positive or negative for the fitness of the given user information and job name.\nUser information: {}\nJob name: {}"
        model_name_or_path = '/scratch2/snail0822/career/job-fitness-qwen-0.5b-pos-neg'
    elif value == 'requirements':
        prompt = "Answer the requirements for the given job.\n{}"
        model_name_or_path = '/scratch2/snail0822/career/job-requirements-7b'
    return prompt, model_name_or_path
    