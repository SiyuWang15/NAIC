## Run Y2H
time=$(date "+%m%d-%H-%M-%S")
echo $time
CUDA_VISIBLE_DEVICES=6 python main.py --runner y2mode --log_prefix $time --Pn 8 &
CUDA_VISIBLE_DEVICES=6 python main.py --runner y2mode --log_prefix $time --Pn 32