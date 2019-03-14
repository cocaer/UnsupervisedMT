import copy

import torch
import torch.nn as nn
import os
import argparse
import subprocess
from src.data.loader import check_all_data_params, load_data
from src.utils import bool_flag, initialize_exp
from src.model import check_mt_model_params, build_mt_model
from src.trainer import TrainerMT
from src.evaluator import EvaluatorMT
import pdb
import random

parser = argparse.ArgumentParser(description='Language transfer')
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")  # 保存文件的地址
parser.add_argument("--dump_path", type=str, default="./dumped/",
                    help="Experiment dump path")
parser.add_argument("--save_periodic", type=bool_flag, default=False,
                    help="Save the model periodically")
parser.add_argument("--seed", type=int, default=-1,
                    help="Random generator seed (-1 for random)")
# autoencoder parameters
parser.add_argument("--emb_dim", type=int, default=512,
                    help="Embedding layer size")
parser.add_argument("--n_enc_layers", type=int, default=4,
                    help="Number of layers in the encoders")
parser.add_argument("--n_dec_layers", type=int, default=4,
                    help="Number of layers in the decoders")
parser.add_argument("--hidden_dim", type=int, default=512,
                    help="Hidden layer size")
parser.add_argument("--lstm_proj", type=bool_flag, default=False,
                    help="Projection layer between decoder LSTM and output layer")
parser.add_argument("--dropout", type=float, default=0,
                    help="Dropout")
parser.add_argument("--label-smoothing", type=float, default=0,
                    help="Label smoothing")
parser.add_argument("--attention", type=bool_flag, default=True,
                    help="Use an attention mechanism")
if not parser.parse_known_args()[0].attention:
    parser.add_argument("--enc_dim", type=int, default=512,
                        help="Latent space dimension")
    parser.add_argument("--proj_mode", type=str, default="last",
                        help="Projection mode (proj / pool / last)")
    parser.add_argument("--init_encoded", type=bool_flag, default=False,
                        help="Initialize the decoder with the encoded state. Append it to each input embedding otherwise.")
else:
    parser.add_argument("--transformer", type=bool_flag, default=True,
                        help="Use transformer architecture + attention mechanism")
    if parser.parse_known_args()[0].transformer:
        parser.add_argument("--transformer_ffn_emb_dim", type=int, default=2048,
                            help="Transformer fully-connected hidden dim size")
        parser.add_argument("--attention_dropout", type=float, default=0,
                            help="attention_dropout")
        parser.add_argument("--relu_dropout", type=float, default=0,
                            help="relu_dropout")
        parser.add_argument("--encoder_attention_heads", type=int, default=8,
                            help="encoder_attention_heads")
        parser.add_argument("--decoder_attention_heads", type=int, default=8,
                            help="decoder_attention_heads")
        parser.add_argument("--encoder_normalize_before", type=bool_flag, default=False,
                            help="encoder_normalize_before")
        parser.add_argument("--decoder_normalize_before", type=bool_flag, default=False,
                            help="decoder_normalize_before")
    else:
        parser.add_argument("--input_feeding", type=bool_flag, default=False,
                            help="Input feeding")
        parser.add_argument("--share_att_proj", type=bool_flag, default=False,
                            help="Share attention projetion layer")
parser.add_argument("--share_lang_emb", type=bool_flag, default=False,
                    help="Share embedding layers between languages (enc / dec / proj)")
parser.add_argument("--share_encdec_emb", type=bool_flag, default=False,
                    help="Share encoder embeddings / decoder embeddings")
parser.add_argument("--share_decpro_emb", type=bool_flag, default=False,
                    help="Share decoder embeddings / decoder output projection")
parser.add_argument("--share_output_emb", type=bool_flag, default=False,
                    help="Share decoder output embeddings")
parser.add_argument("--share_lstm_proj", type=bool_flag, default=False,
                    help="Share projection layer between decoder LSTM and output layer)")
parser.add_argument("--share_enc", type=int, default=0,
                    help="Number of layers to share in the encoders")
parser.add_argument("--share_dec", type=int, default=0,
                    help="Number of layers to share in the decoders")
# encoder input perturbation
parser.add_argument("--word_shuffle", type=float, default=0,
                    help="Randomly shuffle input words (0 to disable)")
parser.add_argument("--word_dropout", type=float, default=0,
                    help="Randomly dropout input words (0 to disable)")
parser.add_argument("--word_blank", type=float, default=0,
                    help="Randomly blank input words (0 to disable)")
# discriminator parameters
parser.add_argument("--dis_layers", type=int, default=3,
                    help="Number of hidden layers in the discriminator")
parser.add_argument("--dis_hidden_dim", type=int, default=128,
                    help="Discriminator hidden layers dimension")
parser.add_argument("--dis_dropout", type=float, default=0,
                    help="Discriminator dropout")
parser.add_argument("--dis_clip", type=float, default=0,
                    help="Clip discriminator weights (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0,
                    help="GAN smooth predictions")
parser.add_argument("--dis_input_proj", type=bool_flag, default=True,
                    help="Feed the discriminator with the projected output (attention only)")
# dataset
parser.add_argument("--langs", type=str, default="",
                    help="Languages (lang1,lang2)")
parser.add_argument("--vocab", type=str, default="",
                    help="Vocabulary (lang1:path1;lang2:path2)")
parser.add_argument("--vocab_min_count", type=int, default=0,
                    help="Vocabulary minimum word count")
parser.add_argument("--mono_dataset", type=str, default="",
                    help="Monolingual dataset (lang1:train1,valid1,test1;lang2:train2,valid2,test2)")
parser.add_argument("--para_dataset", type=str, default="",
                    help="Parallel dataset (lang1-lang2:train12,valid12,test12;lang1-lang3:train13,valid13,test13)")
parser.add_argument("--back_dataset", type=str, default="",
                    help="Back-parallel dataset, with noisy source and clean target (lang1-lang2:train121,train122;lang2-lang1:train212,train211)")
parser.add_argument("--n_mono", type=int, default=0,
                    help="Number of monolingual sentences (-1 for everything)")
parser.add_argument("--n_para", type=int, default=0,
                    help="Number of parallel sentences (-1 for everything)")
parser.add_argument("--n_back", type=int, default=0,
                    help="Number of back-parallel sentences (-1 for everything)")
parser.add_argument("--max_len", type=int, default=175,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--max_vocab", type=int, default=-1,
                    help="Maximum vocabulary size (-1 to disable)")
# training steps
parser.add_argument("--n_dis", type=int, default=0,
                    help="Number of discriminator training iterations")
parser.add_argument("--mono_directions", type=str, default="",
                    help="Training directions (lang1,lang2)")
parser.add_argument("--para_directions", type=str, default="",
                    help="Training directions (lang1-lang2,lang2-lang1)")
parser.add_argument("--pivo_directions", type=str, default="",
                    help="Training directions with online back-translation, using a pivot (lang1-lang3-lang1,lang1-lang3-lang2)]")
parser.add_argument("--back_directions", type=str, default="",
                    help="Training directions with back-translation dataset (lang1-lang2)")
parser.add_argument("--otf_sample", type=float, default=-1,
                    help="Temperature for sampling back-translations (-1 for greedy decoding)")
parser.add_argument("--otf_backprop_temperature", type=float, default=-1,
                    help="Back-propagate through the encoder (-1 to disable, temperature otherwise)")
parser.add_argument("--otf_sync_params_every", type=int, default=1000, metavar="N",
                    help="Number of updates between synchronizing params")
parser.add_argument("--otf_num_processes", type=int, default=30, metavar="N",
                    help="Number of processes to use for OTF generation")
parser.add_argument("--otf_update_enc", type=bool_flag, default=True,
                    help="Update the encoder during back-translation training")
parser.add_argument("--otf_update_dec", type=bool_flag, default=True,
                    help="Update the decoder during back-translation training")
# language model training
parser.add_argument("--lm_before", type=int, default=0,
                    help="Training steps with language model pretraining (0 to disable)")
parser.add_argument("--lm_after", type=int, default=0,
                    help="Keep training the language model during MT training (0 to disable)")
parser.add_argument("--lm_share_enc", type=int, default=0,
                    help="Number of shared LSTM layers in the encoder")
parser.add_argument("--lm_share_dec", type=int, default=0,
                    help="Number of shared LSTM layers in the decoder")
parser.add_argument("--lm_share_emb", type=bool_flag, default=False,
                    help="Share language model lookup tables")
parser.add_argument("--lm_share_proj", type=bool_flag, default=False,
                    help="Share language model projection layers")
# training parameters
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--group_by_size", type=bool_flag, default=True,
                    help="Sort sentences by size during the training")
parser.add_argument("--lambda_xe_mono", type=str, default="0",
                    help="Cross-entropy reconstruction coefficient (autoencoding)")
parser.add_argument("--lambda_xe_para", type=str, default="0",
                    help="Cross-entropy reconstruction coefficient (parallel data)")
parser.add_argument("--lambda_xe_back", type=str, default="0",
                    help="Cross-entropy reconstruction coefficient (back-parallel data)")
parser.add_argument("--lambda_xe_otfd", type=str, default="0",
                    help="Cross-entropy reconstruction coefficient (on-the-fly back-translation parallel data)")
parser.add_argument("--lambda_xe_otfa", type=str, default="0",
                    help="Cross-entropy reconstruction coefficient (on-the-fly back-translation autoencoding data)")
parser.add_argument("--lambda_dis", type=str, default="0",
                    help="Discriminator loss coefficient")
parser.add_argument("--lambda_lm", type=str, default="0",
                    help="Language model loss coefficient")
parser.add_argument("--enc_optimizer", type=str, default="adam,lr=0.0003",
                    help="Encoder optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--dec_optimizer", type=str, default="enc_optimizer",
                    help="Decoder optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--dis_optimizer", type=str, default="rmsprop,lr=0.0005",
                    help="Discriminator optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--clip_grad_norm", type=float, default=5,
                    help="Clip gradients norm (0 to disable)")
parser.add_argument("--epoch_size", type=int, default=100000,
                    help="Epoch size / evaluation frequency")
parser.add_argument("--max_epoch", type=int, default=100000,
                    help="Maximum epoch size")
parser.add_argument("--stopping_criterion", type=str, default="",
                    help="Stopping criterion, and number of non-increase before stopping the experiment")
# reload models
parser.add_argument("--pretrained_emb", type=str, default="",
                    help="Reload pre-trained source and target word embeddings")
parser.add_argument("--pretrained_out", type=bool_flag, default=False,
                    help="Pretrain the decoder output projection matrix")
parser.add_argument("--reload_model", type=str, default="",
                    help="Reload a pre-trained model")
parser.add_argument("--reload_enc", type=bool_flag, default=False,
                    help="Reload a pre-trained encoder")
parser.add_argument("--reload_dec", type=bool_flag, default=False,
                    help="Reload a pre-trained decoder")
parser.add_argument("--reload_dis", type=bool_flag, default=False,
                    help="Reload a pre-trained discriminator")
# freeze network parameters
parser.add_argument("--freeze_enc_emb", type=bool_flag, default=False,
                    help="Freeze encoder embeddings")
parser.add_argument("--freeze_dec_emb", type=bool_flag, default=False,
                    help="Freeze decoder embeddings")
# evaluation
parser.add_argument("--eval_only", type=bool_flag, default=False,
                    help="Only run evaluations")
parser.add_argument("--beam_size", type=int, default=0,
                    help="Beam width (<= 0 means greedy)")
parser.add_argument("--length_penalty", type=float, default=1.0,
                    help="Length penalty: <1.0 favors shorter, >1.0 favors longer sentences")

# copy

parser.add_argument("--exp_name_", type=str, default="",
                    help="Experiment name")
parser.add_argument("--exp_id_", type=str, default="",
                    help="Experiment ID")  # 保存文件的地址
parser.add_argument("--langs_", type=str, default="",
                    help="Languages (lang1,lang2)")
parser.add_argument("--para_directions_", type=str, default="",
                    help="Training directions (lang1-lang2,lang2-lang1)")
parser.add_argument("--pivo_directions_", type=str, default="",
                    help="Training directions with online back-translation, using a pivot (lang1-lang3-lang1,lang1-lang3-lang2)]")

parser.add_argument("--mono_dataset_", type=str, default="",
                    help="Monolingual dataset (lang1:train1,valid1,test1;lang2:train2,valid2,test2)")
parser.add_argument("--pretrained_emb_", type=str, default="",
                    help="Reload pre-trained source and target word embeddings")
parser.add_argument("--para_dataset_", type=str, default="",
                    help="Parallel dataset (lang1-lang2:train12,valid12,test12;lang1-lang3:train13,valid13,test13)")
parser.add_argument("--mono_directions_", type=str, default="",
                    help="Training directions (lang1,lang2)")

parser.add_argument("--fastbpe", type=str, default="",
                    help="path for fastbpe")

parser.add_argument("--bpe_code", type=str, default="",
                    help="path for fastbpe")

parser.add_argument("--fine_tune_directions", type=str, default="",
                    help="path for fastbpe")
parser.add_argument("--word_dico", type=str, default="",
                    help="path for fastbpe")

# lang1 = 'en'
# lang2 = 'ru2en'
# lang3 = 'ru'
# lang4 = 'en2ru'

parser.add_argument("--lang1", type=str, default="en")
parser.add_argument("--lang2", type=str, default="ch2en")
parser.add_argument("--lang3", type=str, default="ch")
parser.add_argument("--lang4", type=str, default="en2ch")

# criterion1 = 'bleu_ru2en_en_test'
# criterion2 = 'bleu_en2ru_ru_test'

parser.add_argument("--criterion1", type=str, default="bleu_ch2en_en_test")
parser.add_argument("--criterion2", type=str, default="bleu_en2ch_ch_test")

params = parser.parse_args()


def load(params):
    # check parameters
    assert params.exp_name
    check_all_data_params(params)
    check_mt_model_params(params)

    # initialize experiment / load data / build model
    data = load_data(params)
    encoder, decoder, discriminator, lm = build_mt_model(params, data)

    # initialize trainer / reload checkpoint / initialize evaluator
    trainer = TrainerMT(encoder, decoder, discriminator, lm, data, params)
    trainer.reload_checkpoint()
    trainer.test_sharing()  # check parameters sharing
    evaluator = EvaluatorMT(trainer, data, params)
    return trainer, evaluator


def get_batch_from_trainer(trainer, lang1, lang2):
    """
    当lang1!=lang2 时，返回平行语料。这里我对进行了封装，返回的不是“en,ch”。而是“en,en2fr”
    当lang2=None是，返回单语语料
    """
    if lang2 is not None:
        (sent1, len1), (sent1_, len1_) = trainer.get_batch('encdec', lang1, lang2)
        return sent1, len1, sent1_, len1_
    return trainer.get_batch('encdec', lang1, None)


def get_fake_para_from_trainer(trainer, lang1, lang2, params):
    """
    example: lang1=fr2en , lang2=en
    :return  fr2en, len(fr2en),  en1, len(en1), en2, len(en2)
    其中en1 是fr2en没有用中文翻译的英文， en2 是翻译结果
    """
    lang1_id = params.lang2id[lang1]
    lang2_id = params.lang2id[lang2]

    sent1, len1, sent1_, len1_ = get_batch_from_trainer(trainer, lang1, lang2)
    with torch.no_grad():
        sent1 = sent1.cuda()
        encoded = trainer.encoder(sent1, len1, lang1_id)
        sent2, len2, _ = trainer.decoder.generate(encoded, lang2_id)

    return sent1, len1, sent1_, len1_, sent2, len2


def get_fake_para_from_mono_trainer(trainer, lang1, lang2, params):
    lang1_id = params.lang2id[lang1]
    lang2_id = params.lang2id[lang2]

    sent1, len1 = get_batch_from_trainer(trainer, lang1, None)
    with torch.no_grad():
        sent1 = sent1.cuda()
        encoded = trainer.encoder(sent1, len1, lang1_id)
        sent2, len2, _ = trainer.decoder.generate(encoded, lang2_id)

    return sent1, len1, _, _, sent2, len2


def convert_to_text(batch, lengths, trainer, lang, remove_bpe=True):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    lang_id = trainer.params.lang2id[lang]
    params = trainer.params
    dico = trainer.data['dico'][lang]
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()
    bos_index = params.bos_index[lang_id]
    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs

    assert (batch[0] == bos_index).sum() == bs
    assert (batch == params.eos_index).sum() == bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sent = " ".join(words)
        if remove_bpe:
            sent = (sent + ' ').replace('@@ ', '').rstrip()
        sentences.append(sent)
    return sentences


def convert_txt_to_tensor(txt, trainer, lang):
    params = trainer.params
    bos_flag = params.bos_index[params.lang2id[lang]]
    dico = trainer.data['dico'][lang]
    txt = [torch.LongTensor([dico.index(em, no_unk=False) for em in s.split()]) if len(s) > 0 else \
               torch.LongTensor([dico.index('unk', no_unk=False)]) \
           for s in txt]

    lengths = torch.LongTensor([s.size(0) + 2 for s in txt])
    sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(dico.pad_index)
    sent[0] = bos_flag
    for i, s in enumerate(txt):
        sent[1:lengths[i] - 1, i].copy_(s)
        sent[lengths[i] - 1, i] = dico.eos_index
    return sent, lengths


def fix_params(params):
    pivot = {}
    params.fine_tune_lang = []
    bpe_code = {}
    word_dico = {}
    for em in params.fine_tune_directions.split(';'):
        pivot[em.split(',')[0]] = em.split(',')[1]
        params.fine_tune_lang.append(em.split(',')[0])

    params.fine_tune_directions = pivot

    for i in range(len(pivot)):
        bpe_code[params.fine_tune_lang[i]] = params.bpe_code.split(',')[i]
        word_dico[params.fine_tune_lang[i]] = params.word_dico.split(',')[i]

    params.bpe_code, params.word_dico = bpe_code, word_dico

    dico = {}
    for lang in params.fine_tune_lang:
        dico[lang] = {}
        with open(word_dico[lang], 'r')  as f:
            for line in f:
                tokens = line.split()
                assert len(tokens) >= 2 and tokens[0] not in dico[lang]
                if tokens[0] not in dico[lang]:
                    dico[lang][tokens[0]] = []
                dico[lang][tokens[0]].append(' '.join(tokens[1:]))  # 一个单词可能有多个释义
    params.word_dico = dico


def dico_translation(txt, dico):
    tokens = [sen.split() for sen in txt]
    re = []
    for sen in tokens:
        sent_split = []
        for em in sen:
            trans = dico.get(em, [em])
            trans = random.choice(trans).split()
            sent_split.extend(trans)
        re.append(' '.join(sent_split))
    return re


def apply_bpe(txt, fastbpe, codes):
    with open('tmp', 'w') as f:
        f.write('\n'.join(txt) + '\n')

    cmd = '{} applybpe tmpbpe tmp {}'.format(fastbpe, codes)
    subprocess.Popen(cmd, shell=True).wait()

    with open('tmpbpe', 'r') as f:
        txt = f.read().splitlines()
    return txt


def convert_lang(sent, length, trainer1, lang1, trainer2, lang2, fastbpe, bpecodes, word_dico=None):
    """
    convert lang1 to lang2
    :param bpecodes:(a)lang1(2)lang2
    :param word_dico:(a)lang1(2)lang2
    :return:
    """
    sent = convert_to_text(sent, length, trainer1, lang1)
    sent = dico_translation(sent, word_dico)

    sent = apply_bpe(sent, fastbpe, bpecodes)
    sent, length = convert_txt_to_tensor(sent, trainer2, lang2)
    return sent, length


def train(trainer, sent1, len1, sent2, len2, lang1, lang2):
    params = trainer.params
    lang1_id = params.lang2id[lang1]
    lang2_id = params.lang2id[lang2]
    # loss_fn = trainer.decoder.loss_fn[lang2_id]
    loss_fn = nn.CrossEntropyLoss()
    n_words = params.n_words[lang2_id]

    encoded = trainer.encoder(sent1, len1, lang1_id)
    scores = trainer.decoder(encoded, sent2[:-1], lang2_id)  # [:-1]表示除了最后 一个元素
    loss = loss_fn(scores.view(-1, n_words), sent2[1:].view(-1))

    if (loss != loss).data.any():
        logger.error("NaN detected")
        exit()

    # optimizer
    trainer.zero_grad(['enc', 'dec'])
    loss.backward()
    trainer.update_params(['enc', 'dec'])


def move_device(trainer, device):
    trainer.encoder.cuda(device)
    trainer.decoder.cuda(device)


if __name__ == '__main__':
    logger = initialize_exp(params)
    fix_params(params)

    params2 = copy.deepcopy(params)
    params2.exp_name, params2.exp_id, params2.langs, params2.para_directions = params2.exp_name_, params2.exp_id_, params2.langs_, params2.para_directions_
    params2.pivo_directions, params2.mono_dataset, params2.pretrained_emb, params2.mono_dataset = params2.pivo_directions_, params2.mono_dataset_, params2.pretrained_emb_, params2.mono_dataset_
    params2.para_dataset, params2.mono_directions = params2.para_dataset_, params2.mono_directions_
    params2.dump_path = 'dumped/' + params2.exp_name + "/" + params2.exp_id

    # lang1 en lang2 fr2en lang3 fr lang4 en2fr
    lang1 = params.lang1
    lang2 = params.lang2
    lang3 = params.lang3
    lang4 = params.lang4

    criterion1 = params.criterion1
    criterion2 = params.criterion2

    trainer1, evaluator1 = load(params)
    trainer2, evaluator2 = load(params2)

    best_bleu1 = 13.20  # evaluator1.run_all_evals(0)[criterion1]
    best_bleu2 = 9.18  # evaluator2.run_all_evals(0)[criterion2]

    update_cirle = 3

    logger.info("###################best bleu############################")
    logger.info("{}:{},{}:{}".format(criterion1, best_bleu1, criterion2, best_bleu2))

    for epoch in range(20):
        for i in range(10000):

            _, _, sent1, len1, sent2, len2 = get_fake_para_from_trainer(trainer2, lang4, lang3, params2)  # en2fr en fr
            sent2, len2 = convert_lang(sent2, len2, trainer2, lang3, trainer1, lang1, params.fastbpe, params.bpe_code[lang2], word_dico=params.word_dico[lang2])
            sent2[0, :] = params.bos_index[params.lang2id[lang2]]
            sent1[0, :] = params.bos_index[params.lang2id[lang1]]
            sent1, sent2 = sent1.cuda(), sent2.cuda()
            train(trainer1, sent2, len2, sent1, len1, lang2, lang1)

            _, _, sent1, len1, sent2, len2 = get_fake_para_from_trainer(trainer1, lang2, lang1, params)  # fr2en fr en
            sent2, len2 = convert_lang(sent2, len2, trainer1, lang1, trainer2, lang3, params.fastbpe, params.bpe_code[lang4], word_dico=params.word_dico[lang4])
            sent2[0] = params.bos_index[params2.lang2id[lang4]]
            sent1[0] = params.bos_index[params2.lang2id[lang3]]
            sent1, sent2 = sent1.cuda(), sent2.cuda()
            train(trainer2, sent2, len2, sent1, len1, lang4, lang3)

            #################################

            sent1, len1, _, _, sent2, len2 = get_fake_para_from_mono_trainer(trainer2, lang4, lang3,
                                                                             params2)  # en2fr  _ fr
            sent2[0, :] = params2.bos_index[params2.lang2id[lang3]]
            sent1[0, :] = params2.bos_index[params2.lang2id[lang4]]
            sent1, sent2 = sent1.cuda(), sent2.cuda()
            train(trainer2, sent2, len2, sent1, len1, lang3, lang4)

            sent1, len1, _, _, sent2, len2 = get_fake_para_from_mono_trainer(trainer2, lang3, lang4, params2)
            sent2[0, :] = params2.bos_index[params2.lang2id[lang4]]
            sent1[0, :] = params2.bos_index[params2.lang2id[lang3]]
            sent1, sent2 = sent1.cuda(), sent2.cuda()
            train(trainer2, sent2, len2, sent1, len1, lang4, lang3)

            sent1, len1, _, _, sent2, len2 = get_fake_para_from_mono_trainer(trainer1, lang2, lang1, params)  # fr2en en
            sent2[0, :] = params.bos_index[params.lang2id[lang1]]
            sent1[0, :] = params.bos_index[params.lang2id[lang2]]
            sent1, sent2 = sent1.cuda(), sent2.cuda()
            train(trainer1, sent2, len2, sent1, len1, lang1, lang2)

            sent1, len1, _, _, sent2, len2 = get_fake_para_from_mono_trainer(trainer1, lang1, lang2, params)  # en fr2en
            sent2[0, :] = params.bos_index[params.lang2id[lang2]]
            sent1[0, :] = params.bos_index[params.lang2id[lang1]]
            sent1, sent2 = sent1.cuda(), sent2.cuda()
            train(trainer1, sent2, len2, sent1, len1, lang2, lang1)

            if i % 100 == 0:
                logger.info("trainning at epoch:{},iter{}".format(epoch, i))
            if i % 500 == 0 and i != 0:
                logger.info("trainning at epoch:{},iter{}".format(epoch, i))
                trainer1.save_checkpoint('_online_'+str(epoch))
                trainer2.save_checkpoint('_online_'+str(epoch))

                bleu1 = evaluator1.run_all_evals(0)[criterion1]
                bleu2 = evaluator2.run_all_evals(0)[criterion2]
                if bleu1 > best_bleu1:
                    trainer1.save_checkpoint(str(bleu1))
                    best_bleu1 = bleu1
                if bleu2 > best_bleu2:
                    best_bleu2 = bleu2
                    trainer2.save_checkpoint(str(bleu2))

