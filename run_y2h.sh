## Run Y2H
time=$(date "+%m%d-%H-%M-%S")
# echo $time
CUDA_VISIBLE_DEVICES=2 python main.py --runner y2hcnn --mode 0 --Pn 8 --log_prefix $time 
# CUDA_VISIBLE_DEVICES=3 python main.py --runner y2hcnn --mode 1 --Pn 8 --log_prefix $time &
# CUDA_VISIBLE_DEVICES=4 python main.py --runner y2hcnn --mode 2 --Pn 8 --log_prefix $time &
# CUDA_VISIBLE_DEVICES=4 python main.py --runner y2hcnn --mode 0 --Pn 32 --log_prefix $time & 
# CUDA_VISIBLE_DEVICES=5 python main.py --runner y2hcnn --mode 1 --Pn 32 --log_prefix $time & 
# CUDA_VISIBLE_DEVICES=5 python main.py --runner y2hcnn  --mode 2 --Pn 32 --log_prefix $time