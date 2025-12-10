import fire
from mylmeval import open_json, save_json, get_results
import random
import os

prompt_1 = """This is about counseling ethics. Based on the following content, describe in single sentences what a good counselor would likely do and what a bad counselor would likely do.
Content: {}
A good counselor:"""

prompt_2 = """The following includes a counseling session and specific counseling ethics. We want to modify the counseling session content to make it a {} counseling. 
Rewrite the session to either adhere to or violate the specific counseling ethics.

Counseling session: {}
Ethically: {}
Counseling ethics: {}"""


def augment(
    model_name_or_path : str = 'mistralai/Mistral-7B-Instruct-v0.3',
    task : str = 'gather_stage2',
    start : int = 0,
    ):
    
    # cactus = [{'inputs' : [c['dialogue']]} for c in cactus]
    
    if task == 'stage1':
        ethics = open_json('data/data9_counseling_ethics.json')
        cactus = open_json('data/cactus.json')
        ethics = [{'inputs' : [v]} for k, v in ethics.items()]
        get_results(
            model_name_or_path=model_name_or_path,
            prompt=prompt_1,
            data=ethics,
            batch_size=len(ethics),
            max_tokens = 256,
            temperature = 0.5,
            save_path='data/mistral0.2_ethics.json',
            do_log=True
            )
        os.remove('data/data9_counseling_ethics.json')
    
    elif task == 'gather_stage1':
        gptgen = open_json('data/mistral0.2_ethics.json')
        real_ethics = []
        for i in range(len(gptgen)):
            original = list(ethics.items())[i]
            try:
                good, bad = gptgen[i]['result'].split("\n\n")
            except:
                good = gptgen[i]['result']
                bad = ''
                print(i)
            real_ethics.append({'key' : original[0], 'value' : original[1], 'counsel' : good, 'type' : 'good'})
            real_ethics.append({'key' : original[0], 'value' : original[1], 'counsel' : bad, 'type' : 'bad'})
            
        save_json(real_ethics, 'data/data9_aca_ethics.json')
        os.remove('data/mistral0.2_ethics.json')
    
    elif task == 'stage2':
        random.seed(42)
        ethics = open_json('data/data9_aca_ethics.json')
        cactus_sampled = random.sample(cactus, len(ethics) * 5)
        input_data = [{'inputs' : [e['type'], c['dialogue'], e['type'], e['counsel']]} for c, e in zip(cactus_sampled, ethics * 5)][start:]
        
        get_results(
            model_name_or_path=model_name_or_path,
            prompt=prompt_2,
            data=input_data,
            batch_size=5,
            max_tokens = 256,
            temperature = 0.5,
            save_path=f'data/ethics_cactus.jsonl',
            do_log=True
            )
    
    
    elif task == 'gather_stage2':
        gptgen = open_json('data/ethics_cactus.jsonl')
        real_ethics = []
        instruction = "This is a dialogue between a counselor and a client. Please identify if the counselor's response is ethical or unethical."
        import pdb; pdb.set_trace()
        for data in gptgen:
            output = data['prompt'].split('Ethically: ')[-1].split('\n')[0]
            rule = data['prompt'].split('Counseling ethics: ')[-1].strip("\n")
            session = [r for r in data['result'].split('\n') if len(r) > 2][1:]
            utter_1 = [r for r in session if 'Client:' in r][0]
            utter_2 = [r for r in session if 'Counselor:' in r][0]
            utter_else = [r for r in session if ('Client:' not in r and 'Counselor:' not in r)][0] # no Client or Counselor
            if len(utter_else) > 0:
                import pdb; pdb.set_trace()
            else:
                for client, counselor in list(zip(utter_1, utter_2)):
                    real_ethics.append({
                        'instruction' : instruction,
                        'rule' : rule,
                        'input' : f"{client}\n{counselor}",
                        'output' : output,
                    })            
        save_json(real_ethics, 'data/data9_aca_ethics_cactus.json')
    


if __name__ == '__main__':
    fire.Fire(augment)