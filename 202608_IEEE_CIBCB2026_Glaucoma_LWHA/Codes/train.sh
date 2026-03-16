
python3 train.py ./ --sched cosine --epochs 100 --warmup-epochs 5 --lr 0.00005 --reprob 0.5 --remode pixel --batch-size 8 --amp -j 4 --num-classes 2 --img-size 224 --opt Adam --dataset csv --train_csv  ./data/LAG/train.txt --val_csv ./data/LAG/val.txt --experiment nf_resnet26_cbam --model nf_resnet26_cbam
