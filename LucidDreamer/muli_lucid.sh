
# Define the first and second commands
cmd1="python run.py --image /data/tandw/waymo_new/segment-3988957004231180266_5566_500_5586_500_with_camera_labels/RGB_image/112_FRONT.png --text /data/xiesr/lucidsim/LucidSim/text.txt --reinpaint"
cmd2="python run.py --image /data/tandw/waymo_new/segment-3988957004231180266_5566_500_5586_500_with_camera_labels/RGB_image/038_FRONT.png --text /data/xiesr/lucidsim/LucidSim/text.txt --reinpaint"

# Loop to run each command 15 times
for i in {1..15}
do
    echo "Running command 1, iteration $i"
    eval $cmd1
    echo "Running command 2, iteration $i"
    eval $cmd2
done

echo "All commands executed successfully."