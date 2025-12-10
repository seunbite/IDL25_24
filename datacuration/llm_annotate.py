import json
from mylmeval import open_json, save_json, get_results

prompt = """다음 일을 하는 직업은 무엇인가요?
설명: {}
직업:"""

def translate_kor_jobdic():
    data = open_json('data/kor_job_dictionary.json')
    
    inputs = []
    for i in range(len(data)):
        job = data[i]['jobname']
        meaning = data[i]['definition']
        essentials = data[i]['explanation']
        inputs.append({'inputs' : [meaning]})
    
    print(len(inputs))
    results = get_results(
        prompt,
        inputs,
        batch_size=10,
        do_save=True,
        save_path='data/mistral0.2_kor_job_dictionary.json',
        do_log=True
        )

    
translate_kor_jobdic()