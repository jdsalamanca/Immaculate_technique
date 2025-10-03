from datetime import datetime
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Union, Any, cast
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import asyncio
import os
from asyncio import Lock
from src.preprocessing_functions import load_video

load_lock = Lock()
config: dict[str, Any] = {"max_num_frames": 512}

@asynccontextmanager
async def lifespan(app: FastAPI):

    #Stuff we want executed upon the app startup
    config["generation_config"] = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0.1,
        num_beams=1
    )
    config["num_segments"] = 128
    yield

def get_technique_review(video_path, num_segments, generation_config, model, tokenizer, exercise):

    with torch.no_grad():
        video_loading_start = datetime.now()
        pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
        print("Video Loading Latency:", datetime.now() - video_loading_start)
        pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

        # single-turn conversation
        start1 = datetime.now()
        if exercise == "squat":
            question = """
                You need to check the form of the squat in the video. comment on the follwoing things:
                1. Feet: Are the heals or toes lifting from the ground.
                2. Knees: Are they collapsing inward
                3. Hips: Are they shooting up in the ascent before the knees or moving to the sides
                4. Back: Is the back rounding during the movement
                5. Depth: Are the hips droping at least to the same level of the knees
                6. Bar path: If visible, is the bar path a straight line through the mid foot
                """
        else:
            question = ""

        input = video_prefix + question
        output, chat_history = model.chat(tokenizer, pixel_values, input, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        print("Latency 1:", datetime.now() - start1)
        question = "Based on the points you described, give the athlete feedback on his form. Be enthusiastic and uplifting"
        review, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)

    return review

app = FastAPI(lifespan=lifespan)

upload_dir = "./videos"
os.makedirs(upload_dir, exist_ok=True)

app.post("/init")
async def init_model():
    try:
        start = datetime.now()
        async with load_lock: 
            #Lazy load of the model and tokenizer
            message = "Model and tokenizer succesfully loaded"
            model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
            
            if "tokenizer" not in config:
                config["tokenizer"] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            else:
                message = "Tokenizer already loaded"
            if "model" not in config:
                config["model"] = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)
            else:
                message += "Model already loaded
                
        return {"Message": message, "Latency": str(datetime.now()-start)}
    except Exception as e:
        return {"Error": f"Error laoding model {str(e)}"}

@app.post("/review")
async def get_review(file: UploadFile=File(...), exercise: str=Form(...)):
    try:
        if "model" not in config and "tokenizer" not in config:
            return {"Error": "Please initialize the mdoel first"}
        video_path = os.path.join(upload_dir, cast(str, file.filename))
        content: bytes = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        #------------------------------------------
        #Potential option:
        #with open(file_path, "wb") as buffer:
         #   shutil.copyfileobj(file.file, buffer)
        #------------------------------------------
        num_segments = config["num_segments"]
        generation_config = config["generation_config"]    
        model = config["model"]
        tokenizer = config["tokenizer"]
        review_task = asyncio.to_thread(get_technique_review, video_path, num_segments, generation_config, model, tokenizer, exercise)
        feedback = await review_task
        # Delete the video to free disk
        os.remove(video_path)
        return {"feedback": feedback}, 200
    except Exception as e:
        return {"Error": str(e)}

@app.post("/test")
async def test_endpoint(message: str=Form(...)):
    return {"config": config["generation_config"], "message": message}
