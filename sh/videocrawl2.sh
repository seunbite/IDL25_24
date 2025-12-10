#!/bin/bash
source ~/.bashrc
conda activate career
cd /scratch2/sb/vtts

csv_file=${1:-"data/yt_urls.csv"}
start_line=${2:-2}
num_lines=${3:-1000}

file_name=$(basename "$csv_file")
sub_lang=$(echo "$file_name" | cut -d'_' -f1)

echo "CSV file: $csv_file"
echo "Start line: $start_line"
echo "Number of lines: $num_lines"
echo "Subtitle language: $sub_lang"

# CSV 파일 존재 확인
if [ ! -f "$csv_file" ]; then
    echo "Error: CSV file not found: $csv_file"
    exit 1
fi

# URL 컬럼 번호 찾기 (첫 번째 줄에서)
url_column=$(head -n 1 "$csv_file" | tr ',' '\n' | grep -n "url" | cut -d':' -f1)
if [ -z "$url_column" ]; then
    echo "Error: 'url' column not found in CSV file"
    exit 1
fi

# 총 처리할 라인 수 계산
total_lines=$(tail -n +$start_line "$csv_file" | head -n $num_lines | wc -l)

# Progress bar function
progress_bar() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local completed=$((percent / 2))
    local remaining=$((50 - completed))
    printf "\rProgress: [%-${completed}s%-${remaining}s] %d%%" "$(printf '%0.s#' $(seq 1 $completed))" "$(printf '%0.s-' $(seq 1 $remaining))" $percent
}

# 지정된 줄부터 시작하여 지정된 개수의 URL에 대해 yt-dlp 실행
current_line=0
tail -n +$start_line "$csv_file" | head -n $num_lines | while IFS=',' read -r url title
do
    current_line=$((current_line + 1))
    if [ -n "$url" ]; then
        echo -e "\nProcessing: $url"
        yt-dlp -x --audio-format mp3 --sub-langs "$sub_lang" --write-auto-subs --skip-download "$url"
    fi
    progress_bar $current_line $total_lines
done


echo -e "\nProcessing complete!"