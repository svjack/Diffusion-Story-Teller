<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Diffusion-Story-Teller</h3>

  <p align="center">
   		A Story Teller supported by Huggingface Inference Api on Stable-Diffusion and LLM 
    <br />
  </p>
</p>

[中文介绍](README.md)

## Brief introduction

### BackGround
Huggingface Inference Api is a free of use toolkit for trying models on the fly, one can try rapid prototyping on AI applications with the help of it. <br/>
This project is an attempt to build a Graphic storytelling project use their api.

### Try Demo on the fly


|Name | HuggingFace Space link |
|---------|--------|
| 🎥💬 Book Cover (Comet Atomic) Story Teller | https://huggingface.co/spaces/svjack/Comet-Atomic-Story-Teller |
| 🧱 Pixel Story Teller | https://huggingface.co/spaces/svjack/Pixel-Story-Teller |

## Installation and Running Results
### Install and Running Step
Install by 
```bash
pip install -r requirements.txt
```
Run Book Cover Story Teller 
```bash
python book_cover_app.py
```
Run Pixel Story Teller 
```bash
python pixel_app.py
```
Then visit 127.0.0.1:7860

### Note 
in above demos, will get and use Huggingface API_TOKEN from environment variables, you can set it mannally.
```python
API_TOKEN = os.environ.get("HF_READ_TOKEN")
```

### Running Results



https://github.com/svjack/Diffusion-Story-Teller/assets/27874014/7f3ce5cb-372c-4256-b27d-522b76709685




https://github.com/svjack/Diffusion-Story-Teller/assets/27874014/ff7b7666-9031-4e09-b75b-455f3ab75ce6



<br/>

Following are some results of two demos.

#### 🎥💬 Book Cover Story Teller

https://github.com/svjack/Diffusion-Story-Teller/assets/27874014/8923961e-adbc-4841-8ca2-fbca714964e9

#### 🧱 Pixel Story Teller

https://github.com/svjack/Diffusion-Story-Teller/assets/27874014/eb99fe76-11a8-4693-804a-55cec36d5968


https://github.com/svjack/Diffusion-Story-Teller/assets/27874014/76c96f9d-8541-4496-9534-b37cbdc65b1a

For more compare results, you can take a look at. [videos](videos)

## Architecture
The story teller can deal with the story of "Someone Do SomeThing", the LLM part complete the cause, process and result,
and the Stable-Diffusion part draw images for them. <br/>
* ’🎥💬 Book Cover Story Teller‘ can add book cover to the story (click from left image gallery), and all the image are transformed to the cover style.
* ‘🧱 Pixel Story Teller’ can downsampling the image to pixel style, make the output like shot from pixel games.

<br/>

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - https://huggingface.co/svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/Diffusion-Story-Teller](https://github.com/svjack/Diffusion-Story-Teller)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Huggingface](https://huggingface.co)
* [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/)
* [svjack](https://huggingface.co/svjack)
