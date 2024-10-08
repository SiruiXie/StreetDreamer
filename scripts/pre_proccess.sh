DATA_DIR="./data/waymo_tfrecords/1.4.2"
OUTPUT_DIR="./data/processed_waymo"

for FILE in "$INPUT_DIR"/*.tfrecord; do
    echo "Proccessing $FILE"
    python ./scripts/pre_process.py \
        --tfrecord_path "$FILE" \
        --output_dir "$OUTPUT_DIR"
done

echo "All files processed."