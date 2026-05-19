#!/bin/bash
# run_onepage_experiments.sh
# Runs one-page experiments across PadL values and monitored pages

PADL_VALUES=(1)
MON_PAGES=(0)
IN_BASE="datasets"

for PADL in "${PADL_VALUES[@]}"; do
    echo "============================================"
    echo "PadL = ${PADL}"
    echo "============================================"

    DEFENDED="${IN_BASE}/CW_tamaraw_padl${PADL}.npz"
    if [ ! -f "$DEFENDED" ]; then
        echo "ERROR: ${DEFENDED} not found. Run run_tamaraw_CW.py with PadL=${PADL} first."
        continue
    fi

    for PAGE in "${MON_PAGES[@]}"; do
        echo "--- PadL=${PADL}, Page=${PAGE} ---"

        DATASET_NAME="CW_tam_p${PADL}_page${PAGE}"
        DATASET_DIR="${IN_BASE}/${DATASET_NAME}"

        # Create balanced one-page dataset
        python make_onepage_dataset.py \
            --in_path ${DEFENDED} \
            --out_dir ${DATASET_DIR} \
            --mon_label ${PAGE}

        # Copy for split script compatibility
        cp ${DATASET_DIR}/onepage.npz ${IN_BASE}/${DATASET_NAME}.npz

        # Split
        python exp/dataset_process/dataset_split.py \
            --dataset ${DATASET_NAME}

        # Generate TAM for RF
        for split in train valid test; do
            python -u exp/dataset_process/gen_tam.py \
                --dataset ${DATASET_NAME} \
                --seq_len 5000 \
                --in_file ${split}
        done

        # DF
        python -u exp/train.py --dataset ${DATASET_NAME} --model DF \
            --device cuda:0 --feature DIR --seq_len 5000 \
            --train_epochs 30 --batch_size 128 --learning_rate 2e-3 \
            --optimizer Adamax \
            --eval_metrics Accuracy Precision Recall F1-score \
            --save_metric F1-score --save_name max_f1

        python -u exp/test.py --dataset ${DATASET_NAME} --model DF \
            --device cuda:0 --feature DIR --seq_len 5000 \
            --batch_size 256 \
            --eval_metrics Accuracy Precision Recall F1-score \
            --load_name max_f1

        # Tik-Tok
        python -u exp/train.py --dataset ${DATASET_NAME} --model TikTok \
            --device cuda:0 --feature DT --seq_len 5000 \
            --train_epochs 30 --batch_size 128 --learning_rate 2e-3 \
            --optimizer Adamax \
            --eval_metrics Accuracy Precision Recall F1-score \
            --save_metric F1-score --save_name max_f1

        python -u exp/test.py --dataset ${DATASET_NAME} --model TikTok \
            --device cuda:0 --feature DT --seq_len 5000 \
            --batch_size 256 \
            --eval_metrics Accuracy Precision Recall F1-score \
            --load_name max_f1

        # Var-CNN
        python -u exp/train.py --dataset ${DATASET_NAME} --model VarCNN \
            --device cuda:0 --feature DT2 --seq_len 5000 \
            --train_epochs 30 --batch_size 50 --learning_rate 1e-3 \
            --optimizer Adam \
            --eval_metrics Accuracy Precision Recall F1-score \
            --save_metric F1-score --save_name max_f1

        python -u exp/test.py --dataset ${DATASET_NAME} --model VarCNN \
            --device cuda:0 --feature DT2 --seq_len 5000 \
            --batch_size 256 \
            --eval_metrics Accuracy Precision Recall F1-score \
            --load_name max_f1

        # RF
        python -u exp/train.py --dataset ${DATASET_NAME} --model RF \
            --device cuda:0 --train_file tam_train --valid_file tam_valid \
            --feature TAM --seq_len 1800 \
            --train_epochs 30 --batch_size 200 --learning_rate 5e-4 \
            --optimizer Adam \
            --eval_metrics Accuracy Precision Recall F1-score \
            --save_metric F1-score --save_name max_f1

        python -u exp/test.py --dataset ${DATASET_NAME} --model RF \
            --device cuda:0 --test_file tam_test \
            --feature TAM --seq_len 1800 \
            --batch_size 256 \
            --eval_metrics Accuracy Precision Recall F1-score \
            --load_name max_f1

    done
done