import subprocess

uids = list(range(0, 50))

for uid in uids:
    cmds = [
        'python', 'kg_okvqa_main.py',
        f'--start={uid * 100}',
        f'--end={(uid+1)*100}',
        f'--test_json_name=test_cot_{uid+1}.json',
        '--model', 'flan-t5-base',
        '--user_msg', 'rationale',
        '--img_type', 'detr',
        '--bs', '2',  # Convert to string
        '--eval_bs', '1',  # Convert to string
        '--eval_acc', '10',  # Convert to string
        '--output_len', '512',  # Convert to string
        '--final_eval',
        '--prompt_format', 'QC-EA',  # Add a comma
        '--evaluate_dir', 'experiments/answer_flan-t5-base_detr_QC-EA_lr5e-05_bs1_op512_ep20/checkpoint-176300'
    ]
    subprocess.run(cmds)
