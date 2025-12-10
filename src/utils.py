import json

def open_json(dir):
    return json.load(open(dir, 'r'))

def save_json(data, dir, n=None):
    if n != None: # time efficiency
        if len(data) % n == 0:
            with open(dir, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(dir, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
                
def flatten(list):
    return [r for t in list for r in t]
