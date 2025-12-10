from mylmeval import open_json, save_json
import random
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import Counter

def content_into_string(content):
    return content['main'] + " " + content.get('detail', "")


def load_diversity(test_size=10000, seed=0, only_main=False, initial_node_idx=1) -> Tuple[List[Dict], Dict]:
    graphs = defaultdict(list)
    data = open_json('data/evalset/diversity.jsonl')
    for item in data:
        graphs[item['idx']].append(item)
    
    graph_ids = list(graphs.keys())
    if initial_node_idx != None:
        testsets = [r for r in graph_ids if len(graphs[r]) > initial_node_idx+2]
    else:
        testsets = graph_ids
        initial_node_idx = 0
    random.seed(seed)
    if test_size:
        test_graph_ids = random.sample(testsets, test_size)
    else:
        test_graph_ids = testsets
    
    testset = [
            {
            'initial_node' : content_into_string(graphs[id][initial_node_idx]['content']) if not only_main else graphs[id][initial_node_idx]['content']['main'], 
            'nodes' : [content_into_string(item['content']) for item in graphs[id][initial_node_idx+1:]], 
            'graph_id' : id,
            'previous_nodes' : [content_into_string(item['content']) for item in graphs[id][:initial_node_idx]],
            'lang' : graphs[id][0]['meta']['lang']
            }
        for id in test_graph_ids
        ]
    
    return testset, graphs


def load_high_qual_diversity(test_size=500, seed=0, only_main=False, initial_node_idx=1, do_keyword=False, keyword='Psychology') -> Tuple[List[Dict], Dict]:
    testset, graphs = load_diversity(test_size=0, seed=seed, only_main=only_main, initial_node_idx=initial_node_idx)
    
    if do_keyword:
        keyword_subset = [r for r in testset if keyword.lower() in ' '.join([k.lower() for k in [r['initial_node']] + r['nodes'] + r['previous_nodes']])]
        random.seed(seed)
        subset = random.sample(keyword_subset, min(test_size, len(keyword_subset)))
    
    else:
        subset = sorted(testset, key=lambda x: len(x['nodes']), reverse=True)[:test_size]
        
    print("subset size:", len(subset))
    print(Counter([len(r['nodes']) for r in subset]))
    return subset, graphs
    

def load_issue(test_size=None, seed=0, only_main=False, graph_version=False) -> Tuple[List[Dict], Dict]:
    en = open_json('data/evalset/issue_en.json')
    ko = open_json('data/evalset/issue_ko.json')
    
    testset = []
    
    def _issue_initial_node(item, only_main, graph_version, lang='en'):
        if graph_version:
            res = [r for r in item['Graph'] if r['type'] == 'issue'][0]['content']
            return content_into_string(res) if not only_main else res['main']
        return item['Post Text'] if lang == 'en' else item['q_content']
    
    for lang, issueset in [('en', en), ('ko', ko)]:
        for i, item in enumerate(issueset):
            testset.append({
                'initial_node' : _issue_initial_node(item, only_main, graph_version, lang),
                'graph_id' : i if lang == 'en' else i+len(en),
                'previous_nodes' : item['Graph'],
                'raw_data' : 
                    {
                        'content' : item['Post Text'] if lang == 'en' else item['q_content'],
                        'title' : item['Title'] if lang == 'en' else item['q_title'],
                        'options' : [r['choice']+". "+r['detail'] for r in item['pathway']],
                    },
                'language' : lang,
            })
    if test_size:
        random.seed(seed)
        testset = random.sample(testset, test_size)

    return testset, None


def load_issue_pathways(test_size=None, seed=0) -> Tuple[List[Dict], Dict]:
    en = open_json('data/evalset/issue_en.json')
    ko = open_json('data/evalset/issue_ko.json')
    
    testset = []
    for i, item in tqdm(enumerate(en)):
        testset.append({
            'title' : item['Title'],
            'issue' : item['Post Text'],
            'options' : [r['choice']+". "+r['detail'] for r in item['pathway']],
            'language' : 'en',
        })
    for i, item in tqdm(enumerate(ko)):
        testset.append({
            'title' : item['q_title'],
            'issue' : item['q_content'],
            'options' : [r['choice']+". "+r['detail'] for r in item['pathway']],
            'language' : 'ko',
        })
    
    return testset, None
