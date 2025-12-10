import pandas as pd
import fire
import os
from adhoc.inspect_transcript import inspect
import datetime

def save_video2dataset_csv(
    csv_file: str,
    ):

    df = pd.read_csv(csv_file, sep='\t')
    dedup_df = df.drop_duplicates(subset=['url'])
    download_path = f'/scratch2/sb/career-ytb-scripts-{os.path.splitext(os.path.basename(csv_file))[0]}-0'
    try:
        done_urls = inspect(download_path)
        print(f"Inspecting {download_path} for done urls", len(done_urls))
    except:
        done_urls = []
    
    dedup_csv_path = csv_file.replace('.csv', '_dedup.csv')
    dedup_df[['url', 'title']].to_csv(dedup_csv_path, sep=',', index=False)
    print(f"De-duplicated CSV saved to {dedup_csv_path}, {len(dedup_df)} / {len(df)}")

    # undone_csv_path = csv_file.replace('.csv', '_undone.csv')
    # undone_df = dedup_df[~dedup_df['url'].isin(done_urls)]
    # undone_df[['url', 'title']].to_csv(undone_csv_path, sep=',', index=False)
    # print(f"Undone CSV saved to {undone_csv_path}")
    # print(f"TOTAL: {len(dedup_df)}, DONE: {len(done_urls)}, UNDONE: {len(undone_df)}")
    
    
if __name__ == "__main__":
    fire.Fire(save_video2dataset_csv)
