video_folder="../data"
audio_folder="../outputs/vta-ldm-clip4clip-v-large"
output_folder="../outputs/merged_video"
# output_folder="outputs/merge_video_youtube_example"
if [ ! -d $output_folder ]; then
    mkdir -p $output_folder
fi
# for video in $video_folder/*; do
# for the first 30 video files
for video in $(ls $video_folder | head -30); do
    video="$video_folder/$video"
    video_name=$(basename $video)
    audio_name=$(basename "$video_name" .mp4)
    # audio_name=$video_name
    audio_name="$audio_name.wav"
    audio_path="$audio_folder/$audio_name"
    echo $audio_path
    if [ -f $audio_path ]; then
        echo "Processing $video_name"
        ffmpeg -y -i $video -i $audio_path -c:a aac -map 0:v:0 -map 1:a:0 $output_folder/$video_name.mkv
        ffmpeg -y -i $output_folder/$video_name.mkv -c:a aac $output_folder/$video_name
        rm $output_folder/$video_name.mkv
    fi
done