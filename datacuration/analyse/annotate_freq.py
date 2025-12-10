from collections import defaultdict
from mylmeval import open_json
import fire
from tqdm import tqdm

majors = """Mechanical Engineering
Computer Science
Business Administration
Psychology
Electrical Engineering
Economics
Biochemistry
Architecture
Political Science
Chemical Engineering
Environmental Science
Industrial Design
International Relations
Marketing
Civil Engineering
Mathematics
Biology
Graphic Design
Sociology
Information Technology
Nursing
Finance
English Literature
Physics
Media Studies
Pharmaceutical Sciences
Music Performance
Anthropology
Data Science
History
Biomedical Engineering
Accounting
Urban Planning
Chemistry
Film Studies
Agricultural
Communication Studies
Software Engineering
Fine Arts
Veterinary Medicine
Linguistics
Materials Science
Hospitality Management
Aerospace Engineering
Public Health
Marine Biology
Fashion Design
Human Resource Management
Statistics
Artificial Intelligence
Dental
Interior Design
Astronomy
Sports Management
Public Policy
Digital Media
Food Science
Theatre Arts
Biotechnology
Real Estate
Gender Studies
Network Engineering
Art History
Clinical Psychology
Supply Chain Management
Microbiology
Dance
Robotics
International Business
Landscape Architecture
Quantum
Game Design
Immunology
Educational Leadership
Forensic Science
Industrial Engineering
Journalism
Neuroscience
Corporate Finance
Sustainable Energy
Photography
Actuarial Science
Physical Therapy
Religious Studies
Information Systems
Geology
Social Work
Animation
Applied Mathematics
Child Development
Risk Management
Molecular Biology
Music Production
Operations Management
Cybersecurity
Philosophy
Wildlife
User Experience Design""".split("\n")


def run(
    file_path: str = 'results/eval_prompt3_30gen_30gen_30gen_30gen_80/Qwen_Qwen2.5-3B-Instruct.jsonl'
    ):
    def count_major(major):
        cnt = 0
        result = []
        for graph_id, graph in graphs.items():
            for item in graph:
                if bool(major in item['content']['main'].lower()+item['content'].get('main', '').lower()):
                    cnt += 1
                    result.append((graph_id, item))
        return cnt, result
    
    graphs = defaultdict(list)
    data = open_json('data/evalset/diversity.jsonl')
    for item in data:
        graphs[item['idx']].append(item)
        
    cnt_results = []
    data = open_json(file_path)
    for item in tqdm(data):
        job_recommendations = [r['content'] for r in item['nodes']]
        for job in job_recommendations:
            cnt, result = count_major(job)
            cnt_results.append((job, cnt, result))
        
    cnt_results.sort(key=lambda x: x[1], reverse=True)
    for item in cnt_results:
        print(item[0], item[1])
        
        
if __name__ == '__main__':
    fire.Fire(run)