#!/bin/bash
shell_dir=$(dirname "$(readlink -f "$0")")
cd "$shell_dir/../video2dataset"

csv_file=${1:-'../data/urls.csv'}
filename=$(basename "$csv_file")
filename_without_extension="${filename%.*}"
current_time=$(date "+%Y-%m%d-%H%M%S")
job_file="${csv_file%.*}_undone.csv"
output_folder="/scratch2/sb/career-ytb-scripts-${filename_without_extension}-0/${current_time}"
el_output_folder="/Users/sb/Downloads/career-ytb-scripts-${filename_without_extension}-0/${current_time}"

echo "Videos will be saved in ${output_folder}"

# Function to check for GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi &> /dev/null
        return $?
    else
        return 1
    fi
}

# Function to perform SCP
scp_files() {
    local src_dir="$1"
    local dest="dc3:{$output_folder}"
    
    find "$src_dir" -type f -print0 | while IFS= read -r -d '' file; do
        scp "$file" "$dest"
        if [ $? -eq 0 ]; then
            echo "Successfully transferred: $file"
            rm "$file"
        else
            echo "Failed to transfer: $file"
        fi
    done
}

# Build the command string


if check_gpu; then
    source ~/.bashrc
    conda activate career
    cmd1="python ../datacuration/urls_to_video2dataset.py --csv_file=\"${csv_file}\""
    eval $cmd1
    echo "GPU detected. Processing locally."
    # cmd="video2dataset --url_list=\"${job_file}\" --url_col=\"url\" --caption_col=\"title\" --output_folder=${output_folder} --config video2dataset/configs/sb_config.yaml --stage whisper"
    cmd="video2dataset --url_list=\"${job_file}\" --url_col=\"url\" --caption_col=\"title\" --output_folder=${output_folder} --config video2dataset/configs/sb_config.yaml"
    eval "$cmd"
else
    source ~/.zshrc
    conda activate 3.11
    echo "No GPU detected. Processing locally and transferring files."
    cmd1="python ../datacuration/urls_to_video2dataset.py --csv_file=\"${csv_file}\""
    eval $cmd1
    # cmd="video2dataset --url_list=\"${job_file}\" --url_col=\"url\" --caption_col=\"title\" --output_folder=${el_output_folder} --config video2dataset/configs/sb_config.yaml --stage whisper"
    cmd="video2dataset --url_list=\"${job_file}\" --url_col=\"url\" --caption_col=\"title\" --output_folder=${el_output_folder} --config video2dataset/configs/sb_config.yaml"
    while IFS= read -r line || [[ -n "$line" ]]; do
        eval "$cmd"
        scp_files "$output_folder"
    done < "$job_file"
fi