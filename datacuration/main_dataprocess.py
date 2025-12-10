from mylmeval import open_json

task1_format = "[Job Recommendation] {}\n"
task2_format = "[Requirements] {}\n"
task3_format = "[Counseling] {}\n"


DATALIST = [
    'data/data1_careernet.json',
    'data/data2_youtube.json',
    'data/data3_reddit.json',
    'data/data4_aihub.json',
    'data/data5_jobdict.json',
    'data/data6_international_jobs.json',
    'data/data7_educaweb.json',
    'data/data8_linkedin.json',
    'data/data9_aca_ethics.json',
    'data/data10_careerinterview.json',
    'data/data11_onet.json',
    'data/data13_kaggle_linkedin.json',
    'data/data14_jp_onet.json',
    'data/data15_kaggle_linkedin2.json',
]

def _process1(text):
    return text


def _process5(r):
    etc = ""
    keys = ["정규교육", "숙련기간", "직무기능", "작업강도", "육체활동", "작업장소", "작업환경", "유사명칭", "관련직업", "자격/면허"]
    for key in keys:
        if r[key] != None:
            etc += f"{key}는 {r[key]}이고, "
    return r['definition'] + r['explanation'] + etc[:-4] + "이다."
    
results = []
for i, data_name in enumerate(DATALIST):
    if 'data1' in data_name:
        data = [r for r in open_json(data_name) if r['age'] in ['대학생', '고등학생(17~19세 청소년)']]
        results.extend([{'input' : r['q_content'], 'output' : _process1(r['a_content'])} for r in data])
    elif 'data2' in data_name:
        pass
    elif 'data3' in data_name:
        pass
    elif 'data4' in data_name:
        pass
    elif 'data5' in data_name:
        results.extend([{'input' : r['jobname']['ko'], 'output' : _process5(r)} for r in open_json(data_name)])
    elif 'data6' in data_name:
        pass
    elif 'data7' in data_name:
        pass
    elif 'data8' in data_name:
        pass
    elif 'data9' in data_name:
        pass
    elif 'data10' in data_name:
        pass
    