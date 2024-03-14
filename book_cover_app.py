import os
import tensorflow as tf
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import numpy as np
import PIL.Image
import gradio as gr
import tensorflow_hub as hub
import matplotlib.pyplot as plt

import gradio as gr
import requests
import io
import random
import os
from PIL import Image, ImageDraw, ImageFont

import pandas as pd
from time import sleep
from tqdm import tqdm

import extcolors
from gradio_client import Client

import cv2
import numpy as np
import glob
import pathlib

API_TOKEN = os.environ.get("HF_READ_TOKEN")

DEFAULT_PROMPT = "X go to Istanbul"
DEFAULT_ROLE = "Superman"
DEFAULT_BOOK_COVER = "book_cover_dir/Blank.png"

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def perform_neural_transfer(content_image_input, style_image_input, hub_module = hub_module):
    content_image = content_image_input.astype(np.float32)[np.newaxis, ...] / 255.
    content_image = tf.image.resize(content_image, (400, 600))
    style_image = style_image_input.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, (256, 256))
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    stylized_image = tensor_to_image(stylized_image)
    content_image_input = tensor_to_image(content_image_input)
    stylized_image = stylized_image.resize(content_image_input.size)
    return stylized_image

list_models = [
    "Pixel-Art-XL",
    "SD-1.5",
    "OpenJourney-V4",
    "Anything-V4",
    "Disney-Pixar-Cartoon",
    "Dalle-3-XL",
]


def generate_txt2img(current_model, prompt, is_negative=False, image_style="None style", steps=50, cfg_scale=7,
                     seed=None, API_TOKEN = API_TOKEN):
    if current_model == "SD-1.5":
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    elif current_model == "OpenJourney-V4":
        API_URL = "https://api-inference.huggingface.co/models/prompthero/openjourney"
    elif current_model == "Anything-V4":
        API_URL = "https://api-inference.huggingface.co/models/xyn-ai/anything-v4.0"
    elif current_model == "Disney-Pixar-Cartoon":
        API_URL = "https://api-inference.huggingface.co/models/stablediffusionapi/disney-pixar-cartoon"
    elif current_model == "Pixel-Art-XL":
        API_URL = "https://api-inference.huggingface.co/models/nerijs/pixel-art-xl"
    elif current_model == "Dalle-3-XL":
        API_URL = "https://api-inference.huggingface.co/models/openskyml/dalle-3-xl"


    #API_TOKEN = os.environ.get("HF_READ_TOKEN")
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    if type(prompt) != type(""):
        prompt = DEFAULT_PROMPT

    if image_style == "None style":
        payload = {
            "inputs": prompt + ", 8k",
            "is_negative": is_negative,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed if seed is not None else random.randint(-1, 2147483647)
        }
    elif image_style == "Cinematic":
        payload = {
            "inputs": prompt + ", realistic, detailed, textured, skin, hair, eyes, by Alex Huguet, Mike Hill, Ian Spriggs, JaeCheol Park, Marek Denko",
            "is_negative": is_negative + ", abstract, cartoon, stylized",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed if seed is not None else random.randint(-1, 2147483647)
        }
    elif image_style == "Digital Art":
        payload = {
            "inputs": prompt + ", faded , vintage , nostalgic , by Jose Villa , Elizabeth Messina , Ryan Brenizer , Jonas Peterson , Jasmine Star",
            "is_negative": is_negative + ", sharp , modern , bright",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed if seed is not None else random.randint(-1, 2147483647)
        }
    elif image_style == "Portrait":
        payload = {
            "inputs": prompt + ", soft light, sharp, exposure blend, medium shot, bokeh, (hdr:1.4), high contrast, (cinematic, teal and orange:0.85), (muted colors, dim colors, soothing tones:1.3), low saturation, (hyperdetailed:1.2), (noir:0.4), (natural skin texture, hyperrealism, soft light, sharp:1.2)",
            "is_negative": is_negative,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed if seed is not None else random.randint(-1, 2147483647)
        }

    image_bytes = requests.post(API_URL, headers=headers, json=payload).content
    image = Image.open(io.BytesIO(image_bytes))
    return image

from huggingface_hub import InferenceClient
import gradio as gr
import pandas as pd
import numpy as np
import os

event_reasoning_df = pd.DataFrame(
                [['Use the following events as a background to answer questions related to the cause and effect of time.', 'Ok'],

                ['What are the necessary preconditions for the next event?：X had a big meal.', 'X placed an order'],
                ['What could happen after the next event?：X had a big meal.', 'X becomes fat'],
                ['What is the motivation for the next event?：X had a big meal.', 'X is hungry'],
                ['What are your feelings after the following event?：X had a big meal.', "X tastes good"],

                ['What are the necessary preconditions for the next event?：X met his favorite star.', 'X bought a ticket'],
                ['What could happen after the next event?：X met his favorite star.', 'X is motivated'],
                ['What is the motivation for the next event?：X met his favorite star.', 'X wants to have some entertainment'],
                ['What are your feelings after the following event?：X met his favorite star.', "X is in a happy mood"],

                ['What are the necessary preconditions for the next event?: X to cheat', 'X has evil intentions'],
                ['What could happen after the next event?：X to cheat', 'X is accused'],
                ['What is the motivation for the next event?：X to cheat', 'X wants to get something for nothing'],
                ['What are your feelings after the following event?：X to cheat', "X is starving and freezing in prison"],

                ['What could happen after the next event?：X go to Istanbul', ''],
                             ],
                             columns = ["User", "Assistant"]
                             )

Mistral_7B_client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.1",
    token = API_TOKEN
)

NEED_PREFIX = 'What are the necessary preconditions for the next event?'
EFFECT_PREFIX = 'What could happen after the next event?'
INTENT_PREFIX = 'What is the motivation for the next event?'
REACT_PREFIX = 'What are your feelings after the following event?'

def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

def generate(
    prompt, history, client = Mistral_7B_client,
    temperature=0.7, max_new_tokens=256, top_p=0.95, repetition_penalty=1.1,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output

hist = event_reasoning_df.iloc[:-1, :].apply(
    lambda x: (x["User"], x["Assistant"]), axis = 1
)

def produce_4_event(event_fact, hist = hist):
    NEED_PREFIX_prompt = "{}：{}".format(NEED_PREFIX, event_fact)
    EFFECT_PREFIX_prompt = "{}：{}".format(EFFECT_PREFIX, event_fact)
    INTENT_PREFIX_prompt = "{}：{}".format(INTENT_PREFIX, event_fact)
    REACT_PREFIX_prompt = "{}：{}".format(REACT_PREFIX, event_fact)
    NEED_PREFIX_output = list(generate(NEED_PREFIX_prompt, history = hist, max_new_tokens = 2048))[-1]
    EFFECT_PREFIX_output = list(generate(EFFECT_PREFIX_prompt, history = hist, max_new_tokens = 2048))[-1]
    INTENT_PREFIX_output = list(generate(INTENT_PREFIX_prompt, history = hist, max_new_tokens = 2048))[-1]
    REACT_PREFIX_output = list(generate(REACT_PREFIX_prompt, history = hist, max_new_tokens = 2048))[-1]
    NEED_PREFIX_output, EFFECT_PREFIX_output, INTENT_PREFIX_output, REACT_PREFIX_output = map(lambda x: x.replace("</s>", ""), [NEED_PREFIX_output, EFFECT_PREFIX_output, INTENT_PREFIX_output, REACT_PREFIX_output])
    return {
        NEED_PREFIX: NEED_PREFIX_output,
        EFFECT_PREFIX: EFFECT_PREFIX_output,
        INTENT_PREFIX: INTENT_PREFIX_output,
        REACT_PREFIX: REACT_PREFIX_output,
    }

def transform_4_event_as_sd_prompts(event_fact ,event_reasoning_dict, role_name = "superman"):
    req = {}
    for k, v in event_reasoning_dict.items():
        if type(role_name) == type("") and role_name.strip():
            v_ = v.replace("X", role_name)
        else:
            v_ = v
        req[k] = list(generate("Transform this as a prompt in stable diffusion: {}".\
        format(v_),
              history = [], max_new_tokens = 2048))[-1].replace("</s>", "")
    event_fact_ = event_fact.replace("X", role_name)
    req["EVENT_FACT"] = list(generate("Transform this as a prompt in stable diffusion: {}".\
    format(event_fact_),
          history = [], max_new_tokens = 2048))[-1].replace("</s>", "")
    req_list = [
        req[INTENT_PREFIX], req[NEED_PREFIX],
            req["EVENT_FACT"],
        req[REACT_PREFIX], req[EFFECT_PREFIX]
    ]
    caption_list = [
        event_reasoning_dict[INTENT_PREFIX], event_reasoning_dict[NEED_PREFIX],
            event_fact,
        event_reasoning_dict[REACT_PREFIX], event_reasoning_dict[EFFECT_PREFIX]
    ]
    caption_list = list(map(lambda x: x.replace("X", role_name), caption_list))
    return caption_list ,req_list

def batch_as_list(input_, batch_size = 3):
    req = []
    for ele in input_:
        if not req or len(req[-1]) >= batch_size:
            req.append([ele])
        else:
            req[-1].append(ele)
    return req

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def add_caption_on_image(input_image, caption, marg_ratio = 0.15, row_token_num = 6):
    from uuid import uuid1
    assert hasattr(input_image, "save")
    max_image_size = max(input_image.size)
    marg_size = int(marg_ratio * max_image_size)
    colors, pixel_count = extcolors.extract_from_image(input_image)
    input_image = add_margin(input_image, marg_size, 0, 0, marg_size, colors[0][0])
    font = ImageFont.truetype("DejaVuSerif-Italic.ttf" ,int(marg_size / 4))
    caption_token_list = list(map(lambda x: x.strip() ,caption.split(" ")))
    caption_list = list(map(" ".join ,batch_as_list(caption_token_list, row_token_num)))
    draw = ImageDraw.Draw(input_image)
    for line_num ,line_caption in enumerate(caption_list):
        position = (
        int(marg_size / 4) * (line_num + 1) * 1.1 ,
        (int(marg_size / 4) * (
            (line_num + 1) * 1.1
        )))
        draw.text(position, line_caption, fill="black", font = font)
    return input_image


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height)))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width)))
        return result

def generate_video(images, video_name = 'ppt.avi'):
    import cv2
    from uuid import uuid1
    im_names = []
    for im in images:
        name = "{}.png".format(uuid1())
        im.save(name)
        im_names.append(name)
    frame = cv2.imread(im_names[0])

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    # Appending the images to the video one by one
    for name in im_names:
        video.write(cv2.imread(name))
        os.remove(name)

    # Deallocating memories taken for window creation
    #cv2.destroyAllWindows()
    video.release()  # releasing the video generated

def make_video_from_image_list(image_list, video_name = "ppt.avi"):
    if os.path.exists(video_name):
        os.remove(video_name)
    assert all(map(lambda x: hasattr(x, "save"), image_list))
    max_size = list(map(max ,zip(*map(lambda x: x.size, image_list))))
    max_size = max(max_size)
    image_list = list(map(lambda x: expand2square(x,
                                                 extcolors.extract_from_image(x)[0][0][0]
                                                 ).resize((max_size, max_size)), image_list))

    generate_video(image_list, video_name = video_name)
    return video_name

def style_transfer_func(content_img, style_img):
    assert hasattr(content_img, "save")
    assert hasattr(style_img, "save")
    colors, pixel_count = extcolors.extract_from_image(style_img)
    if colors and colors[0][0] == (255, 255, 255) and (colors[0][1] / sum(map(lambda t2: t2[1] ,colors)) > 0.95):
        return content_img
    content_image_input = np.asarray(content_img)
    style_image_input = np.asarray(style_img)
    out = perform_neural_transfer(content_image_input, style_image_input)
    assert hasattr(out, "save")
    return out


def gen_images_from_event_fact(current_model, event_fact = DEFAULT_PROMPT, role_name = DEFAULT_ROLE,
    style_pic = None
):
    event_reasoning_dict = produce_4_event(event_fact)
    caption_list ,event_reasoning_sd_list = transform_4_event_as_sd_prompts(event_fact ,
        event_reasoning_dict,
        role_name = role_name
    )
    img_list = []
    for prompt in tqdm(event_reasoning_sd_list):
        im = generate_txt2img(current_model, prompt, is_negative=False, image_style="None style")
        img_list.append(im)
        sleep(2)
    img_list = list(filter(lambda x: hasattr(x, "save"), img_list))
    if style_pic is not None and hasattr(style_pic, "size"):
        style_pic = Image.fromarray(style_pic.astype(np.uint8))
        print("perform styling.....")
        img_list_ = []
        for x in tqdm(img_list):
            img_list_.append(style_transfer_func(x, style_pic))
        img_list = img_list_
    img_list = list(map(lambda t2: add_caption_on_image(t2[0], t2[1]) ,zip(*[img_list, caption_list])))
    img_mid = img_list[2]
    img_list_reordered = [img_mid]
    for ele in img_list:
        if ele not in img_list_reordered:
            img_list_reordered.append(ele)
    video_path = make_video_from_image_list(img_list_reordered)
    return video_path

def image_click(images, evt: gr.SelectData,
    ):
    img_selected = images[evt.index][0]["name"]
    return img_selected

def get_book_covers():
    covers = pd.Series(
    list(pathlib.Path("book_cover_dir").rglob("*.jpg")) + \
    list(pathlib.Path("book_cover_dir").rglob("*.png")) + \
    list(pathlib.Path("book_cover_dir").rglob("*.jpeg"))
    ).map(str).map(lambda x: np.nan if x.split("/")[-1].startswith("_") else x).dropna().map(
        lambda x: (x, "".join(x.split(".")[:-1]).split("/")[-1])
    ).values.tolist()
    covers = sorted(covers, key = lambda t2: int(DEFAULT_BOOK_COVER in t2[0]), reverse = True)
    return covers

with gr.Blocks(css=".caption-label {display:none}") as demo:
    favicon = '<img src="" width="48px" style="display: inline">'
    gr.Markdown(
        f"""<h1><center> 🎥💬 Book Cover Story Teller</center></h1>
            """
    )
    with gr.Row():
        with gr.Column(elem_id="prompt-container"):
            current_model = gr.Dropdown(label="Current Model", choices=list_models, value="Pixel-Art-XL")
            style_reference_input_gallery = gr.Gallery(get_book_covers(),
                            height = 768 + (768 - 512) + 64 + 32,
                            label = "StoryBook Cover (click to use)",
                            object_fit = "contain"
                            )
        with gr.Column(elem_id="prompt-container"):
            style_reference_input_image = gr.Image(
                            label = "StoryBook Cover (you can upload yourself or click from left gallery)",
                            value = DEFAULT_BOOK_COVER,
                            interactive = True,
                            )
            with gr.Row():
                role_name = gr.Textbox(label="Role (X)", placeholder=DEFAULT_ROLE, lines=1,
                    elem_id="prompt-text-input", value = DEFAULT_ROLE,
                    info = "You should set the Role (X) with some famous man (like: Confucius Superman)"
                    )
                text_prompt = gr.Textbox(label="Event Prompt", placeholder=DEFAULT_PROMPT,
                    lines=1, elem_id="prompt-text-input", value = DEFAULT_PROMPT,
                    info = "You should set the prompt in format 'X do something', X is the role in the left."
                    )
            with gr.Row():
                text_button = gr.Button("Generate", variant='primary', elem_id="gen-button")

            with gr.Row():
                video_output = gr.Video(label = "Story Video", elem_id="gallery", height = 768,)

    style_reference_input_gallery.select(
            image_click, style_reference_input_gallery, style_reference_input_image
    )

    text_button.click(gen_images_from_event_fact, inputs=[current_model, text_prompt, role_name, style_reference_input_image],
        outputs=video_output)

    gr.Examples(
        [
            ["Pixel-Art-XL", "book_cover_dir/Blank.png", "Confucius", "X read a book"],
            ["Pixel-Art-XL", "book_cover_dir/Blank.png", "Superman", "X go to Istanbul"],
            ["SD-1.5", "book_cover_dir/zh_ancient_tree.jpg", "Monk Xuanzang", "X went to the West to obtain Buddhist scriptures"],
            ["SD-1.5", "book_cover_dir/Blank.png", "Mickey Mouse", "X attends a party"],
            ["SD-1.5", "book_cover_dir/River.png", "Napoleon", "X riding a horse"],
            ["SD-1.5", "book_cover_dir/seaport.png", "The Pope", "X is being crowned"],
        ],
        inputs = [current_model, style_reference_input_image, role_name, text_prompt],
        label = "Example collection 1"
    )

    gr.Examples(
        [
            ["Disney-Pixar-Cartoon", "book_cover_dir/Blank.png", "Harry Potter", "X defeated Voldemort"],
            ["Disney-Pixar-Cartoon", "book_cover_dir/Blank.png", "Minions", "X join the interstellar war"],
            ["OpenJourney-V4", "book_cover_dir/A tale of two cities.jpg", "Augustus Octavian", "X served as tribune"],
            ["SD-1.5", "book_cover_dir/an illustration of a city.jpg", "The Eastern Roman Emperor", "X defeats Mongol Invaders"],
        ],
        inputs = [current_model, style_reference_input_image, role_name, text_prompt],
        label = "Example collection 2"
    )

demo.launch(show_api=False, share=False)
