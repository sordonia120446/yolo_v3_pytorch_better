DATA_DIR='data/pascal_data'

echo 'Get The Pascal VOC Data'
wget -N --directory-prefix=$DATA_DIR https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget -N --directory-prefix=$DATA_DIR https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget -N --directory-prefix=$DATA_DIR https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

echo 'Un-tar the things. Do not use verbose, since it is lots of files'
tar xf "$DATA_DIR/VOCtrainval_11-May-2012.tar"
tar xf "$DATA_DIR/VOCtrainval_06-Nov-2007.tar"
tar xf "$DATA_DIR/VOCtest_06-Nov-2007.tar"

echo "Moving un-tarred things back into $DATA_DIR...tar just is poop"
find . -type f -name "2012_*.txt" -exec mv -t "$DATA_DIR/" {} +
find . -type f -name "2007_*.txt" -exec mv -t "$DATA_DIR/" {} +

echo 'Generate Labels for VOC'
wget -N --directory-prefix=$DATA_DIR http://pjreddie.com/media/files/voc_label.py
python "$DATA_DIR/voc_better_label.py"
cat "$DATA_DIR/2007_train.txt" > "$DATA_DIR/voc_train.txt"
cat "$DATA_DIR/2007_val.txt" >> "$DATA_DIR/voc_train.txt"
cat "$DATA_DIR/2012_train.txt" >> "$DATA_DIR/voc_train.txt"
cat "$DATA_DIR/2012_val.txt" >> "$DATA_DIR/voc_train.txt"

echo "Labeling complete. Check $DATA_DIR to make sure everything is stored there correctly."
