video_paths="../data/video_processed/video_gt_augment"
save_dir="../data/video_processed/audio_gt_augment"

# get wav audio from video
for video_path in $video_paths/*; do
    video_name=$(basename $video_path)
    audio_name="${video_name%.*}.wav"
    audio_path="$save_dir/$audio_name"
    ffmpeg -i $video_path -vn -acodec pcm_s16le -ar 16000 -ac 1 $audio_path
done