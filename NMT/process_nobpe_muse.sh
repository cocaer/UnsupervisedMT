#!/usr/bin/env bash

#!/usr/bin/env bash


#bash process_nobpe.sh  vocab_size data_dir lang1 lang2

VOCAB_SIZE=$1     # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=20      # number of fastText epochs

TOOLS_PATH=$PWD/tools
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

DATA_PATH=$2
SRC_TOK=$DATA_PATH/train.$3.tok
TGT_TOK=$DATA_PATH/train.$4.tok

SRC_VALID=$DATA_PATH/valid.$3.tok
TGT_VALID=$DATA_PATH/valid.$4.tok
SRC_TEST=$DATA_PATH/test.$3.tok
TGT_TEST=$DATA_PATH/test.$4.tok

SRC_VOCAB=$DATA_PATH/$3.vocab
TGT_VOCAB=$DATA_PATH/$4.vocab







echo
echo "-----------------------extract vocabulary--------------------"
$FASTBPE getvocab  $SRC_TOK | head  -35000 > $SRC_VOCAB
$FASTBPE getvocab  $TGT_TOK | head  -35000 > $TGT_VOCAB


# binarize data
echo
echo "-----------------------binarize data--------------------"
python preprocess.py $SRC_VOCAB  $SRC_TOK
python preprocess.py $TGT_VOCAB  $TGT_TOK


python preprocess.py $SRC_VOCAB  $SRC_VALID
python preprocess.py $TGT_VOCAB  $TGT_VALID


python preprocess.py $SRC_VOCAB  $SRC_TEST
python preprocess.py $TGT_VOCAB  $TGT_TEST

#$FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $SRC_TOK -output $SRC_TOK
#$FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $TGT_TOK -output $TGT_TOK
