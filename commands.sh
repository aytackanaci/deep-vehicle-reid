WEIGHTS=e__S_veri776-aic19track2_T_aic19track2_mpfl_E__warmup0_\[40\]_m60_P_b32_adam_lr0.001_wd0.0005_idTrue_orientTrue_landmarksTrue/checkpoint_ep54.pth.tar
OPTIONS='-s veri776 aic19track2 -t aic19track2 --root ~/Data/ --keypoints-dir /home/georgia/Data/VehicleReIDKeyPointData/ 0 --aic19-manual-labels --visualize-ranks --lr 0.001 --optim adam --train-batch-size 32 --use-avai-gpus -j 8'
A_OPTIONS='-s veri776 aic19track2 -t aic19track2 --root ~/Data/ --aic19-manual-labels --visualize-ranks --lr 0.001 --optim adam --train-batch-size 32 --use-avai-gpus -j 8'

# Resume Train
#python3 train_imgreid_mpfl.py $OPTIONS --resume=$WEIGHTS --eval-freq 1 --max-epoch 100

# Train
#python3 train_imgreid_mpfl.py $OPTIONS --eval-freq 1

# Test
#python3 train_imgreid_mpfl.py $OPTIONS --evaluate --load-weights=$WEIGHTS

# Train Aytac's model
python3 train_imgreid_dpfl.py $A_OPTIONS --eval-freq 1
