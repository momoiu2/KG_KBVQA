	CUDA_VISIBLE_DEVICES=0 python kg_okvqa_main.py \
    --model flan-t5-base\
    --user_msg answer --img_type detr \
    --bs 1 --eval_bs 1 --eval_acc 10 --output_len 512  --input_len 512 --epoch 20 --lr 5e-5\
    --final_eval --prompt_format QC-EA \