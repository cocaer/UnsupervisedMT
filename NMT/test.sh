#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0   python main_gpu.py --exp_name test \
                --transformer True --n_enc_layers 4 --n_dec_layers 4 \
                 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True \
                  --langs 'ch,en2ch' --n_mono -1\
                  --mono_dataset 'ch:/data4/bjji/data/umt_data/umt_en2ch_facebook_nobpe/train.ch.tok.pth,,;en2ch:/data4/bjji/data/umt_data/umt_en2ch_facebook_nobpe/train.en2ch.tok.pth,,'\
                   --para_dataset 'ch-en2ch:,/data4/bjji/data/umt_data/umt_en2ch_facebook_nobpe/valid.XX.tok.pth,/data4/bjji/data/umt_data/umt_en2ch_facebook_nobpe/test.XX.tok.pth' \
                   --mono_directions 'ch,en2ch' --word_shuffle 3 --word_dropout 0.1 \
                   --word_blank 0.2 --pivo_directions 'ch-en2ch-ch,en2ch-ch-en2ch'\
                   --lambda_xe_mono '0:0,100000:0,200000:0' --lambda_xe_otfd 1 --otf_num_processes 5 \
                   --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001   \
                   --epoch_size 500000 --stopping_criterion bleu_en2ch_ch_valid,10  --batch_size 32


	#--pretrained_emb '/data4/bjji/data/umt_data/umt_en2ch_facebook_nobpe//concat.tok.vec'  --pretrained_out True\