# An example script to evaluate Wani2v, gt_path is the path to the ground truth trajectory directory estimated by monst3r, pred_path is the path to the predicted trajectory directory
echo "Evaluating wani2v"

echo "Relative"
python eval_ct.py \
    --gt_path /mnt/sdb/wangxinran/monst3r/original_ct \
    --pred_path /mnt/sdb/wangxinran/monst3r/wani2v_ct \
    --batch \
    --aligment_way downsample \
    --mode relative \
    --save_path wani2v_relative_results.json

echo "Absolute"
python eval_ct.py \
    --gt_path /mnt/sdb/wangxinran/monst3r/original_ct \
    --pred_path /mnt/sdb/wangxinran/monst3r/wani2v_ct \
    --batch \
    --aligment_way downsample \
    --mode absolute \
    --save_path wani2v_absolute_results.json
echo "Done"





