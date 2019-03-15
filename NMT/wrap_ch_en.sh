#!/usr/bin/env bash


MONO_DATASET='ch2en:/data4/bjji/data/umt_data/nobpe/umt_ch2en_facebook_nobpe/train.ch2en.tok.pth,,;en:/data4/bjji/data/umt_data/nobpe/umt_ch2en_facebook_nobpe/train.en.tok.pth,,'
PARA_DATASET='ch2en-en:/data4/bjji/data/umt_data/nobpe/umt_ch2en_facebook_nobpe/fake_para/train.XX.tok.pth,/data4/bjji/data/umt_data/nobpe/umt_ch2en_facebook_nobpe/valid.XX.tok.pth,/data4/bjji/data/umt_data/nobpe/umt_ch2en_facebook_nobpe/test.XX.tok.pth'
PIVO_DIRECTIONS='ch2en-en-ch2en,en-ch2en-en'
PRETRAINED_EMB='/data4/bjji/data/umt_data/nobpe/umt_ch2en_facebook_nobpe/concat.tok.vec'


MONO_DATASET_='ch:/data4/bjji/data/umt_data/nobpe/umt_en2ch_facebook_nobpe//train.ch.tok.pth,,;en2ch:/data4/bjji/data/umt_data/nobpe/umt_en2ch_facebook_nobpe/train.en2ch.tok.pth,,'
PARA_DATASET_='ch-en2ch:/data4/bjji/data/umt_data/nobpe/umt_en2ch_facebook_nobpe/fake_para/train.XX.tok.pth,/data4/bjji/data/umt_data/nobpe/umt_en2ch_facebook_nobpe/valid.XX.tok.pth,/data4/bjji/data/umt_data/nobpe/umt_en2ch_facebook_nobpe/test.XX.tok.pth'
PRETRAINED_EMB_='/data4/bjji/data/umt_data/nobpe/umt_en2ch_facebook_nobpe/concat.tok.vec'
PIVO_DIRECTIONS_='ch-en2ch-ch,en2ch-ch-en2ch'






CUDA_VISIBLE_DEVICES=4  python modelWrapper.py	--dump_path /data4/bjji/data/umt_model/ --exp_name nobpe --exp_id umt_ch2en_facebook_nobpe --eval_only True\
                                            --transformer True \
                                            --n_enc_layers 4 \
                                            --n_dec_layers 4 \
                                            --share_enc 3 \
                                            --share_dec 3 \
                                            --share_lang_emb True \
                                            --share_output_emb True \
                                            --langs 'ch2en,en' \
                                            --n_mono -1 \
                                            --n_para -1\
                                            --mono_dataset  $MONO_DATASET \
                                            --para_dataset  $PARA_DATASET\
                                            --pivo_directions  $PIVO_DIRECTIONS\
                                            --pretrained_emb $PRETRAINED_EMB \
                                            --mono_directions 'ch2en,en' \
                                            --pretrained_out True \
                                            --lambda_xe_otfd 1 \
                                            --otf_num_processes 1 \
                                            --otf_sync_params_every 1000 \
                                            --enc_optimizer adam,lr=0.0001 \
                                            --epoch_size 50000 \
                                            --lambda_xe_mono '0:1,100000:0.1,300000:0' \
                                            --stopping_criterion bleu_ch2en_en_valid,10\
                                            --batch_size 32 --max_len 130   \
                                            --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 \
                                            --exp_name_ nobpe  --exp_id_ umt_en2ch_facebook_nobpe \
                                            --dump_path_ /data4/bjji/data/umt_model/\
                                            --langs_ 'ch,en2ch' \
                                            --mono_dataset_  $MONO_DATASET_ \
                                            --para_dataset_  $PARA_DATASET_\
                                            --pivo_directions_  $PIVO_DIRECTIONS_\
                                            --pretrained_emb_ $PRETRAINED_EMB_ \
                                            --mono_directions_ 'ch,en2ch'\
                                            --fastbpe ""\
                                            --word_dico '/data4/bjji/data/umt_data/bpe/umt_ch2en_facebook_fasttext_bpe/ch2en/zh-en.sim.txt.uniq,/data4/bjji/data/umt_data/bpe/umt_ch2en_facebook_fasttext_bpe/ch2en/en-zh.sim.unique.txt'\
                                            --fine_tune_directions 'ch2en,en;en2ch,ch'\
                                            --bpe_code '/data4/bjji/data/umt_ch2en_facebook/ch2en/bpe_codes,/data4/bjji/data/umt_ch2en_facebook/en2ch/bpe_codes' \
                                            --lang1 en --lang2 ch2en --lang3 ch --lang4 en2ch \
                                            --criterion1  bleu_ch2en_en_test \
                                            --criterion2  bleu_en2ch_ch_test
