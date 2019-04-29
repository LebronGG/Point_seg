i=6
#python batch_inference.py --model_path log${i}/epoch_95.ckpt --dump_dir log${i}/dump --output_filelist log${i}/output_filelist.txt --room_data_filelist meta/area${i}_data_label.txt --visu
python eval_iou_accuracy.py --test_area ${i}