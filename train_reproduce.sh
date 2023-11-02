source envs/res8_robin.env
export NOISE_DATASET_PATH=/home/nlpintern1/howl-ws/sampled_noise/
echo $LR_DECAY
LR_DECAY=0.98 VOCAB='["hello","cape","bee"]' \
USE_NOISE_DATASET=True BATCH_SIZE=1 \
INFERENCE_THRESHOLD=0 NUM_EPOCHS=300 \
NUM_MELS=40 \
INFERENCE_SEQUENCE=[0,1,2] \
MAX_WINDOW_SIZE_SECONDS=0.5 \
python -W ignore -m training.run.train \
--dataset-paths /home/nlpintern1/howl-ws/howl/datasets/hello_cape_bee/positive /home/nlpintern1/howl-ws/howl/datasets/hello_cape_bee/negative \
--model res8 \
--workspace /home/nlpintern1/howl-ws/howl/workspaces/hello-cape-bee-res8-reproduce-neg-talknoise \
--use-stitched-datasets