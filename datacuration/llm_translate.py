import json
from mylmeval import open_json, save_json, get_results, cal_token, cal_price
import fire



prompt = """Please translate this {} job name into {}.
Just give me translated text, without any extra words.
Don't use the same words in the original text, try to translate into target language preserving the meaning.

Original Text: {} ({})
Translated Text:"""

def translate(
    from_language : str = 'Korean',
    to_language : str = 'Japanese',
    cal_token : bool = False,
    model_name_or_path : str = 'mistralai/Mistral-7B-Instruct-v0.3',
    start : int = 0
    ):
    data = open_json('data/data5_jobdict_ko.json')
    
    inputs = []
    for i in range(start, len(data)):
        job = data[i]['jobname']
        description = data[i]['definition']
        inputs.append({'inputs' : [from_language, to_language, job, description]})

    if cal_token:
        output_example = 'Restaurant Manager'
        print("input tokens:", cal_price(
            sum([cal_token(prompt.format(*item['inputs'])) for item in inputs]))
              )
        print("output tokens:", cal_price(sum([cal_token(output_example) for _ in inputs])))
        return
    
    print(len(inputs))
    _ = get_results(
        model_name_or_path=model_name_or_path,
        prompt=prompt,
        data=inputs,
        batch_size=len(data) if 'gpt' not in model_name_or_path else 1,
        temperature=0,
        max_tokens=20,
        save_path=f'data/data5_jobdict_{to_language.lower()[:2]}_{start}.jsonl',
        do_log=True,
        )


def gather_result():
    def _parsing(text):
        return text.split('\n')[0].split("(")[0].strip(" ")
    
    en = open_json('data/data5_jobdict_en.jsonl')
    sp = open_json('data/data5_jobdict_sp.jsonl')
    ja = open_json('data/data5_jobdict_ja.jsonl')
    
    result = []
    original = open_json('data/data5_jobdict_ko.json')
    
    for i, item in enumerate(original):
        item['jobname'] = {
            'ko' : item['jobname'], 
            'en' : _parsing(en[i]['result']), 
            'sp' : _parsing(sp[i]['result']), 
            'ja' : _parsing(ja[i]['result'])
        }
        print(item['jobname'])
        result.append(item)
    
    with open('data/data5_jobdict.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("Done!")
        
        
if __name__ == '__main__':
    # fire.Fire(translate)
    gather_result()