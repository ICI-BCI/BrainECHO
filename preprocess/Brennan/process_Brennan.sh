python preprocess/Brennan/asr_pipeline.py --input_dir /home/szx/dataset/Brennan/audio \
--output_dir preprocess/audio
python data_processing_split_by_shuffle.py
python data_processing_split_by_subject.py
python data_processing_split_by_sents.py