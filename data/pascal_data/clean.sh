DATA_DIR='data/pascal_data'

echo "Removing data files in $DATA_DIR"

find $DATA_DIR -type f -name "*.tar" -delete
find $DATA_DIR -type f -name "2007_*.txt" -delete
find $DATA_DIR -type f -name "2012_*.txt" -delete
find $DATA_DIR -type f -name "voc_train.txt" -delete
find $DATA_DIR -type d -name "VOCdevkit" -exec rm -rf "{}" \;
exit 0
