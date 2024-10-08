DEVICES=0
IMG=/path/to/your/conditioning/img
TEXT=""
CAMERA_PATH=autodrivce
RENDER_CAMERA_PAth=autodrive_render
CONTROLNET_PATH=../personalized_checkpoint_output/controlnet
UNET_PATH=../personalized_checkpoint_output/unet
SAVE_DIR=../dreamer_result

cd ../LucidDreamer

python run.py --image $IMG \
   --controlnet_path $CONTROLNET_PATH \
   --unet_path $UNET_PATH \
   --campath_gen $CAMERA_PATH \
   --campath_render $RENDER_CAMERA_PAth\
   --save_dir $SAVE_DIR \
   --reinpaint
