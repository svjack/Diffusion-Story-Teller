<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Diffusion-Story-Teller</h3>

  <p align="center">
   	调用 Stable-Diffusion 及 LLM 的 Huggingface Inference Api 构建的故事讲述工程 
    <br />
  </p>
</p>

[In English](README_EN.md)

## 简要介绍

### 背景
Huggingface Inference Api 是一个可以免费试用的模型测试工具包，助力快速原型设计人工智能应用。<br/>
本项目旨在利用他们的api构建一套简单的图形化故事呈现系统。

### 即时演示


|名称 | HuggingFace Space 链接 |
|---|---|
| 🎥💬 封面故事讲述者（Comet Atomic） | https://huggingface.co/spaces/svjack/Comet-Atomic-Story-Teller |
| 🧱 像素故事讲述者 | https://huggingface.co/spaces/svjack/Pixel-Story-Teller |

## 安装和运行结果
### 安装和运行步骤
通过命令行安装
```bash
pip install -r requirements.txt
```
运行封面故事讲述者
```bash
python book_cover_app.py
```
运行像素故事讲述者
```bash
python pixel_app.py
```
然后访问 http://127.0.0.1:7860

### 注意
上述演示demo会从环境变量获取Huggingface API\_TOKEN，手动设置也可以。
```python
API_TOKEN = os.environ.get("HF_READ_TOKEN")
```
### 运行结果

https://github.com/svjack/Diffusion-Story-Teller/assets/27874014/7f3ce5cb-372c-4256-b27d-522b76709685


https://github.com/svjack/Diffusion-Story-Teller/assets/27874014/ff7b7666-9031-4e09-b75b-455f3ab75ce6

<br>

以下是两个demo的部分结果。

#### 🎥💬 封面故事讲述者

https://github.com/svjack/Diffusion-Story-Teller/assets/27874014/8923961e-adbc-4841-8ca2-fbca714964e9

#### 🧱 像素故事讲述者

https://github.com/svjack/Diffusion-Story-Teller/assets/27874014/eb99fe76-11a8-4693-804a-55cec36d5968


https://github.com/svjack/Diffusion-Story-Teller/assets/27874014/76c96f9d-8541-4496-9534-b37cbdc65b1a

更多对比结果，请查看[视频](videos)。

## 体系结构
故事讲述者可以根据“某人做某事”的主题生成故事，LLM部分补全故事的起承转合（起因、经过、结果等），而Stable-Diffusion部分则为它们绘制图片。<br/>
* 🎥💬 封面故事讲述者 可以向故事添加书籍封面(点击左侧画廊中的图片)，并且所有图像都转换为封面样式。
* 🧱 像素故事讲述者 可以降低图像分辨率到像素级别，使输出类似像素游戏中的截屏效果。

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
