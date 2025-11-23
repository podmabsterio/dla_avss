if [ $# -lt 1 ]; then
    echo "Error: Please provide an output directory."
    echo "Usage: $0 /path/to/output_directory"
    exit 1
fi

OUTPUT_PATH="$1"
mkdir -p "$OUTPUT_PATH"

FILE_ID="18TEetAQ1212HoMBdnDWMd-1soRHJghA_"
FILE_NAME="convtasnet.pth"

if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing via pip..."
    pip install gdown
fi

echo "Downloading weights..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUTPUT_PATH/$FILE_NAME"

if [ $? -eq 0 ]; then
    echo "Download completed: $OUTPUT_PATH/$FILE_NAME"
else
    echo "Download failed."
fi
