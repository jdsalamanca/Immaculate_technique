from datetime import datetime
import torch
from transformers import AutoModel, AutoTokenizer
from src.preprocessing_functions import load_video

# evaluation setting
max_num_frames = 512
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)
video_path = "/content/drive/MyDrive/Powerlifting/Videos/IMG_0746.MOV"
num_segments=128
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)

def get_technique_review(video_path, generation_config):

    with torch.no_grad():
        video_loading_start = datetime.now()
        pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
        print("Video Loading Latency:", datetime.now() - video_loading_start)
        pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

        # single-turn conversation
        start1 = datetime.now()
        question = """
              You need to check the form of the squat in the video. comment on the follwoing things:
              1. Feet: Are the heals or toes lifting from the ground.
              2. Knees: Are they collapsing inward
              3. Hips: Are they shooting up in the ascent before the knees or moving to the sides
              4. Back: Is the back rounding during the movement
              5. Depth: Are the hips droping at least to the same level of the knees
              6. Bar path: If visible, is the bar path a straight line through the mid foot
              """
        input = video_prefix + question
        output, chat_history = model.chat(tokenizer, pixel_values, input, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        print("Latency 1:", datetime.now() - start1)
        question = "Based on the points you described, give the athlete feedback on his form. Be enthusiastic and uplifting"
        review, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)

    return review

if __name__ == "__main__":
    