## Brennan dataset
# autoencoding
python train_decoding_eeg_2_text_Brennan_Gwilliams_dataset_mel_vqvae_pretrain.py -c config/Brennan_dataset_train_mel_vqvae_pretrain.yaml --dataset Brennan

# alignment and finetuning: split by subject
python train_decoding_eeg_2_text_Brennan_dataset.py -c config/Brennan_dataset_train.yaml \
--dataset_path /home/szx/eeg2text/datasets/Brennan/split_by_subject/ \
--save_path ./checkpoints/whisper_decoding/Brennan_dataset/Brennan_dataset_split_by_subject-4040-1e-4

python eval_decoding_eeg_2_text_Brennan_dataset.py -c config/Brennan_dataset_test.yaml \
--dataset_path /home/szx/eeg2text/datasets/Brennan/split_by_subject/ \
--eeg2text_checkpoint ./checkpoints/whisper_decoding/Brennan_dataset/Brennan_dataset_split_by_subject-4040-1e-4/b16_40_0.0001_finetune.pt \
--text_rusults Brennan_dataset_split_by_subject_result

# alignment and finetuning: split by sentence
python train_decoding_eeg_2_text_Brennan_dataset.py -c config/Brennan_dataset_train.yaml \
--dataset_path /home/szx/eeg2text/datasets/Brennan/split_by_sents/ \
--save_path ./checkpoints/whisper_decoding/Brennan_dataset/Brennan_dataset_split_by_sents-4040-1e-4

python eval_decoding_eeg_2_text_Brennan_dataset.py -c config/Brennan_dataset_test.yaml \
--dataset_path /home/szx/eeg2text/datasets/Brennan/split_by_sents/ \
--eeg2text_checkpoint ./checkpoints/whisper_decoding/Brennan_dataset/Brennan_dataset_split_by_sents-4040-1e-4/b16_40_0.0001_finetune.pt \
--text_rusults Brennan_dataset_split_by_sents_result

## GWilliams dataset
# autoencoding
python train_decoding_eeg_2_text_Brennan_Gwilliams_dataset_mel_vqvae_pretrain.py -c config/Gwilliams_dataset_train_mel_vqvae_pretrain.yaml --dataset gwilliams

# alignment and finetuning: split after random shuffling
python train_decoding_meg_2_text_Gwilliams_dataset.py -c config/Gwilliams_dataset_train.yaml --split split1 \
--save_path ./checkpoints/whisper_decoding/meg_dataset/meg_dataset-split1-4040-1e-4

python eval_decoding_meg_2_text_gwilliams_dataset.py -c config/Gwilliams_dataset_test.yaml \
--eeg2text_checkpoint ./checkpoints/whisper_decoding/meg_dataset/meg_dataset-split1-4040-1e-4/b16_40_0.0002_finetune.pt \
--text_rusults meg_dataset_split1_result

# alignment and finetuning: split by session
python train_decoding_meg_2_text_Gwilliams_dataset.py -c config/Gwilliams_dataset_train.yaml --split split2 \
--save_path ./checkpoints/whisper_decoding/meg_dataset/meg_dataset-split2-4040-1e-4

python eval_decoding_meg_2_text_gwilliams_dataset.py -c config/Gwilliams_dataset_test.yaml \
--eeg2text_checkpoint ./checkpoints/whisper_decoding/meg_dataset/meg_dataset-split2-4040-1e-4/b16_40_0.0002_finetune.pt \
--text_rusults meg_dataset_split2_result

# alignment and finetuning: split by sentence
python train_decoding_meg_2_text_Gwilliams_dataset.py -c config/Gwilliams_dataset_train.yaml --split split3 \
--save_path ./checkpoints/whisper_decoding/meg_dataset/meg_dataset-split3-4040-1e-4

python eval_decoding_meg_2_text_gwilliams_dataset.py -c config/Gwilliams_dataset_test.yaml \
--eeg2text_checkpoint ./checkpoints/whisper_decoding/meg_dataset/meg_dataset-split3-4040-1e-4/b16_40_0.0002_finetune.pt \
--text_rusults meg_dataset_split3_result

# alignment and finetuning: split by subject
python train_decoding_meg_2_text_Gwilliams_dataset.py -c config/Gwilliams_dataset_train.yaml --split split4 \
--save_path ./checkpoints/whisper_decoding/meg_dataset/meg_dataset-split4-4040-1e-4

python eval_decoding_meg_2_text_gwilliams_dataset.py -c config/Gwilliams_dataset_test.yaml \
--eeg2text_checkpoint ./checkpoints/whisper_decoding/meg_dataset/meg_dataset-split4-4040-1e-4/b16_40_0.0002_finetune.pt \
--text_rusults meg_dataset_split4_result
