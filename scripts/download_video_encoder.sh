if [ $# -lt 1 ]; then
    echo "Error: Please provide an output directory."
    echo "Usage: $0 /path/to/output_directory"
    exit 1
fi

OUTPUT_PATH="$1"
mkdir -p "$OUTPUT_PATH"

FILE_ID="1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm"
FILE_NAME="lrw_resnet18_dctcn_video_boundary.pth"

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
