time=$(date "+%m%d-%H-%M-%S")

# CUDA_VISIBLE_DEVICES=6 python main.py --runner validation --log_prefix $time --Pn 32 &
CUDA_VISIBLE_DEVICES=3 python main.py --runner ensemble --Pn 8 --time $time