#!/usr/bin/env bash

#!/usr/bin/env bash
##!/usr/bin/env bash



MONO_DATASET='en:/data/bjji/umt_data/umt_fr2en/all.en.tok.50000.pth,,;fr2en:/data/bjji/umt_data/umt_fr2en/all.fr2en.tok.50000.pth,,'
PARA_DATASET='en-fr2en:/data/bjji/umt_data/umt_fr2en/fr2en/all.XX.tok.50000.pth,/data/bjji/umt_data/umt_fr2en/valid.XX.tok.50000.pth,/data/bjji/umt_data/umt_fr2en/test.XX.tok.50000.pth'
PIVO_DIRECTIONS='fr2en-en-fr2en,en-fr2en-en'
PRETRAINED_EMB='/data/bjji/umt_data/umt_fr2en/concat.bpe.vec'


MONO_DATASET_='fr:/data/bjji/umt_data/umt_en2fr/all.fr.tok.50000.pth,,;en2fr:/data/bjji/umt_data/umt_en2fr/all.en2fr.tok.50000.pth,,'
PARA_DATASET_='en2fr-fr:/data/bjji/umt_data/umt_en2fr/en2fr/all.XX.tok.50000.pth,/data/bjji/umt_data/umt_en2fr/valid.XX.tok.50000.pth,/data/bjji/umt_data/umt_en2fr/test.XX.tok.50000.pth'
PRETRAINED_EMB_='/data/bjji/umt_data/umt_en2fr/concat.bpe.vec'
PIVO_DIRECTIONS_='en2fr-fr-en2fr,fr-en2fr-fr'






CUDA_VISIBLE_DEVICES=2 python modelWrapper.py	--exp_name test --exp_id fr2en_en1 --eval_only True\
                                            --transformer True \
                                            --n_enc_layers 4 \
                                            --n_dec_layers 4 \
                                            --share_enc 3 \
                                            --share_dec 3 \
                                            --share_lang_emb True \
                                            --share_output_emb True \
                                            --langs 'en,fr2en' \
                                            --n_mono -1 \
                                            --n_para -1\
                                            --mono_dataset  $MONO_DATASET \
                                            --para_dataset  $PARA_DATASET\
                                            --pivo_directions  $PIVO_DIRECTIONS\
                                            --pretrained_emb $PRETRAINED_EMB \
                                            --mono_directions 'en,fr2en' \
                                            --pretrained_out True \
                                            --lambda_xe_otfd 1 \
                                            --otf_num_processes 1 \
                                            --otf_sync_params_every 1000 \
                                            --enc_optimizer adam,lr=0.0001 \
                                            --epoch_size 50000 \
                                            --lambda_xe_mono '0:1,100000:0.1,300000:0' \
                                            --stopping_criterion bleu_fr2en_en_valid,10\
                                            --batch_size 40  --max_len 80\
                                            --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 \
                                            --exp_name_ test --exp_id_ en2fr_fr1 \
                                            --langs_ 'en2fr,fr' \
                                            --mono_dataset_  $MONO_DATASET_ \
                                            --para_dataset_  $PARA_DATASET_\
                                            --pivo_directions_  $PIVO_DIRECTIONS_\
                                            --pretrained_emb_ $PRETRAINED_EMB_ \
                                            --mono_directions_ 'en2fr,fr'\
                                            --fastbpe /home/bjji/source/umt_origin/tools/fastbpe/fastBPE/fast\
                                            --word_dico '/data/bjji/umt_data/umt_fr2en/fr-en.txt.unique,/data/bjji/umt_data/umt_en2fr/en-fr.txt.uniq'\
                                            --fine_tune_directions 'fr2en,en;en2fr,fr'\
                                            --bpe_code '/data/bjji/umt_data/umt_fr2en/bpe_codes,/data/bjji/umt_data/umt_en2fr/bpe_codes'







