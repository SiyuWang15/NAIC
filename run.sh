## Run Y2H
# time=$(date "+%m%d-%H-%M-%S")
# echo $time
# CUDA_VISIBLE_DEVICES=3 python main.py --mode 0 --Pn 8 --log_prefix $time & 
# CUDA_VISIBLE_DEVICES=3 python main.py --mode 1 --Pn 8 --log_prefix $time 
# CUDA_VISIBLE_DEVICES=4 python main.py --mode 2 --Pn 8 --log_prefix $time &
# CUDA_VISIBLE_DEVICES=4 python main.py --mode 0 --Pn 32 --log_prefix $time & 
# CUDA_VISIBLE_DEVICES=5 python main.py --mode 1 --Pn 32 --log_prefix $time & 
# CUDA_VISIBLE_DEVICES=5 python main.py --mode 2 --Pn 32 --log_prefix $time

CUDA_VISIBLE_DEVICES=3 python main.py 