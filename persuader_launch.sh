
model_path=###

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2.5-72B-Instruct --model $model_path --port 8088 --tensor-parallel-size 4 --gpu-memory-utilization 0.85 --disable-custom-all-reduce
