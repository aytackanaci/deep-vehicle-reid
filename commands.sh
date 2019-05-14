WEIGHTS=checkpoint_osg_veri_aic_ep100.pth.tar
#WEIGHTS=resnet50_exps/e__S_veri776-boxcars21k-compcars_reid-aic19track2_T_aic19track2-veri776_mpfl_resnet50_lr0.0001_o_s_g_alttrain_fcs1024_xent_250419/checkpoint_ep50.pth.tar
# VeRi+AIC
OPTIONS='-s veri776 aic19track2 -t innovateuk --root ~/Data/ --visualize-ranks --lr 0.0001 --optim adam --train-batch-size 8 --use-avai-gpus -j 8 --max-epoch 200'
# All data
#OPTIONS='-s veri776 boxcars21k compcars_reid vehicleID aic19track2 -t aic19track2 veri776 --root ~/Data/ --keypoints-dir /home/georgia/Data/VehicleReIDKeyPointData/ 0 0 0 0 --visualize-ranks --lr 0.0001 --optim adam --train-batch-size 16 --use-avai-gpus -j 8 --max-epoch 300'
# VeRi only
#OPTIONS='-s veri776 -t veri776 --root ~/Data/ --keypoints-dir /home/georgia/Data/VehicleReIDKeyPointData/ --visualize-ranks --lr 0.0001 --optim adam --train-batch-size 8 --use-avai-gpus -j 8 --max-epoch 200'
# Vehicle ID + VeRi
#OPTIONS='-s veri776 vehicleID -t veri776 vehicleID --root ~/Data/ --keypoints-dir /home/georgia/Data/VehicleReIDKeyPointData/ 0 --visualize-ranks --lr 0.0001 --optim adam --train-batch-size 16 --use-avai-gpus -j 8 --max-epoch 200'
A_OPTIONS='-s veri776 aic19track2 -t aic19track2 --root ~/Data/ --aic19-manual-labels --visualize-ranks --lr 0.001 --optim adam --train-batch-size 32 --use-avai-gpus -j 8'

# Resume Train
#python3 train_imgreid_mpfl.py $OPTIONS --resume=$WEIGHTS --eval-freq 5 #--max-epoch 100

# Train
#python3 train_imgreid_mpfl.py $OPTIONS --eval-freq 1 #--regress-landmarks

# Test
python3 train_imgreid_mpfl.py $OPTIONS --evaluate --load-weights=$WEIGHTS --test-batch-size 500

# Train Aytac's model
#python3 train_imgreid_dpfl.py $A_OPTIONS --eval-freq 1
