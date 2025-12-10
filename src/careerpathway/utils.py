from typing import List, Tuple, Dict
import re
import pandas as pd
import numpy as np
from careerpathway.multiprompt import PROMPTS
import json


def open_json(file_path: str) -> List[Dict]:
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    else:
        with open(file_path, 'r') as f:
            return json.load(f)

def save_json(data: List[Dict], file_path: str, mode: str = 'w'):
    if file_path.endswith('.jsonl'):
        with open(file_path, mode) as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        with open(file_path, mode) as f:
            json.dump(data, f, indent=4)


def load_names_dict(
    name_type: str = 'Localized Name',
    file_dir: str = '/home/iyy1112/workspace/popular-names-by-country-dataset/common-forenames-by-country.csv'
    ): # Localized Name or Romanized Name
    country_names = {}
    df2 = pd.read_csv(file_dir)
    for country in df2['Country'].unique():
        F = df2[(df2['Country'] == country) & (df2['Gender'] == 'F')][name_type].tolist()
        M = df2[(df2['Country'] == country) & (df2['Gender'] == 'M')][name_type].tolist()
        if len(F) > 10 and len(M) > 10:
            country_names[country.lower()] = {'F': F, 'M': M}
    
    if name_type == 'Localized Name':
        country_names['cn'] = {'M' : ['伟', '军', '毅', '刚', '浩', '明', '杰', '峰', '磊', '涛'], 'F' : ['娜', '丽', '霞', '娟', '艳', '红', '敏', '英', '梅', '兰']}
        country_names['kr'] = {'M' : ['민수', '종호', '승훈', '지훈', '동현', '영호', '재우', '성진', '민호', '태영'], 'F' : ['지영', '민정', '수진', '은정', '혜진', '미정', '영희', '지혜', '은영', '미숙']}
        country_names['th'] = {'M' : ['สมชาย', 'ธนวัฒน์', 'กิตติพงศ์', 'ณัฐพล', 'วิชัย', 'อนุวัฒน์', 'ประพัฒน์', 'สุรศักดิ์', 'พิชัย', 'ศักดิ์ชัย'], 'F' : ['สุดา', 'รัตนา', 'วันดี', 'พิมพ์', 'มาลี', 'สมหญิง', 'นภา', 'กุลธิดา', 'สุพร', 'อรุณี']}
        country_names['vi'] = {'M' : ['Minh Quân', 'Đức Anh', 'Hoàng Long', 'Thanh Tùng', 'Quang Vinh', 'Hữu Phát', 'Công Danh', 'Bảo Nam', 'Tuấn Anh', 'Việt Hoàng'], 'F' : ['Thùy Linh', 'Ngọc Anh', 'Phương Anh', 'Mai Hương', 'Thanh Hà', 'Bích Ngọc', 'Minh Tâm', 'Thu Hằng', 'Lan Anh', 'Hồng Nhung']}
        country_names['it'] = {'M' : ['Alessandro', 'Francesco', 'Lorenzo', 'Mattia', 'Matteo', 'Andrea', 'Gabriele', 'Davide', 'Simone', 'Riccardo'], 'F' : ['Giulia', 'Sofia', 'Aurora', 'Alice', 'Greta', 'Emma', 'Giorgia', 'Martina', 'Chiara', 'Francesca']}
        country_names['de'] = {'M' : ['Maximilian', 'Alexander', 'Paul', 'Leon', 'Lukas', 'Felix', 'Jonas', 'Luis', 'Simon', 'Julian'], 'F' : ['Sophie', 'Marie', 'Maria', 'Laura', 'Anna', 'Lena', 'Lea', 'Hannah', 'Lina', 'Emma']}

    elif name_type == 'Romanized Name':
        country_names['cn'] = {'M' : ['Wei', 'Jun', 'Yi', 'Gang', 'Hao', 'Ming', 'Jie', 'Feng', 'Lei', 'Tao'], 'F' : ['Na', 'Li', 'Xia', 'Juan', 'Yan', 'Hong', 'Min', 'Ying', 'Mei', 'Lan']}
        country_names['kr'] = {'M' : ['Minsoo', 'Jongho', 'Seunghun', 'Jihoon', 'Donghyun', 'Youngho', 'Jaewoo', 'Sungjin', 'Minho', 'Taeyoung'], 'F' : ['Jiyoung', 'Minjung', 'Soojin', 'Eunjung', 'Hyejin', 'Mijung', 'Younghee', 'Jihye', 'Eunyoung', 'Misook']}
        country_names['th'] = {'M' : ['Somchai', 'Thanawat', 'Kittipong', 'Nattapon', 'Wichai', 'Anuwat', 'Prapat', 'Surasak', 'Pichai', 'Sakchai'], 'F' : ['Suda', 'Ratana', 'Wandee', 'Pim', 'Malee', 'Somying', 'Napa', 'Kulthida', 'Suporn', 'Arunee']}                                                                                                                            
        country_names['vi'] = {'M' : ['Minh Quan', 'Duc Anh', 'Hoang Long', 'Thanh Tung', 'Quang Vinh', 'Huu Phat', 'Cong Danh', 'Bao Nam', 'Tuan Anh', 'Viet Hoang'], 'F' : ['Thuy Linh', 'Ngoc Anh', 'Phuong Anh', 'Mai Huong', 'Thanh Ha', 'Bich Ngoc', 'Minh Tam', 'Thu Hang', 'Lan Anh', 'Hong Nhung']}
        country_names['it'] = {'M' : ['Alessandro', 'Francesco', 'Lorenzo', 'Mattia', 'Matteo', 'Andrea', 'Gabriele', 'Davide', 'Simone', 'Riccardo'], 'F' : ['Giulia', 'Sofia', 'Aurora', 'Alice', 'Greta', 'Emma', 'Giorgia', 'Martina', 'Chiara', 'Francesca']}
        country_names['de'] = {'M' : ['Maximilian', 'Alexander', 'Paul', 'Leon', 'Lukas', 'Felix', 'Jonas', 'Luis', 'Simon', 'Julian'], 'F' : ['Sophie', 'Marie', 'Maria', 'Laura', 'Anna', 'Lena', 'Lea', 'Hannah', 'Lina', 'Emma']}

    return country_names


def get_random_name(nation: str, sex: str) -> str:
    country_names = load_names_dict()
    return np.random.choice(country_names[nation.lower()][sex.upper()])
    


def extract_num(text: str) -> int:
    num = re.findall(r'\d+', text)
    return int(num[0]) if num else None


def parse_jobs(generation: str, job_n: int = 10) -> List[str]:
    for i in range(job_n, 0, -1):
        generation = generation.replace(f"Job {i}", '')
        generation = generation.replace(f"{i}. ", '')
    generation = generation.replace(": ", "").replace("*", "")
    generation = generation.split('\n')
    return [r for r in generation if len(r) > 2 and 'Here' not in r]


def parse_jobs_and_skills(generation: str, job_n: int = 10, country_code: str = 'us') -> List[str]:
    # 사용자가 입력한 country_code에 해당하는 언어 프롬프트를 선택합니다.
    prompts = PROMPTS
    prompt = prompts.get(country_code, prompts['us'])  # 기본값은 'us'로 설정
    
    # 'generation'을 줄 단위로 분할하여 결과 필터링
    results = [r for r in generation.split("\n") if len(r) > 2 and '|' in r]
    output = []

    # 각 결과 처리
    for result in results:
        parsing_output = {}
        parsing = result.split("|")

        if len(parsing) == 4:
            parsing_output['position'] = parsing[0].split(":")[-1].strip()
            parsing_output['salary'] = parsing[1].split(":")[-1].strip()
            parsing_output['year'] = extract_num(parsing[2].split(":")[-1].strip())
            parsing_output['requirements'] = parsing[3].split(":")[-1].strip()
        
        # 'position'이 있는 경우만 결과에 추가
        if 'position' in parsing_output:
            output.append(parsing_output)
    
    # 프롬프트에 맞게 결과 포맷을 반환
    return output


def parse_jobs_and_skills_ko(generation: str, job_n: int = 10) -> List[Dict[str, str]]:
    results = [r for r in generation.split("\n") if '직업' in r]
    output = []
    
    for result in results:
        parsing_output = {}
        parsing = result.split("|")
        
        # 안전하게 각 필드 처리
        if len(parsing) > 0 and '직업' in parsing[0]:
            parsing_output['position'] = parsing[0].split(":")[-1].strip()
        if len(parsing) > 1 and '예상 연봉' in parsing[1]:
            parsing_output['salary'] = parsing[1].split(":")[-1].strip()
        if len(parsing) > 2 and '요구 경력' in parsing[2]:
            parsing_output['year'] = extract_num(parsing[2].split(":")[-1].strip())
        if len(parsing) > 3 and 'Key Skills' in parsing[3]:
            parsing_output['requirements'] = parsing[3].split(":")[-1].strip()
            
        # 최소한 position이 있는 경우만 추가
        if parsing_output['position']:
            output.append(parsing_output)
            
    return output