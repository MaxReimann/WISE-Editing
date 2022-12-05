---
title: White-box Style Transfer Editing (WISE)
emoji: 🎨
colorFrom: pink
colorTo: red
sdk: streamlit
sdk_version: 1.10.0
app_file: Whitebox_style_transfer.py
tags: [Style Transfer,Image Synthesis,Editing,Painting]
pinned: false
license: mit
---
# White-box Style Transfer Editing (WISE) Demo

This app demonstrates the editing capabilities of the [White-box Style Transfer Editing (WISE) framework](https://github.com/winfried-ripken/wise).
It optimizes the parameters of classical image processing filters to match a given style image.
After optimization, parameters can be tuned by hand to achieve a desired look.


### How does it work?
We provide a small stylization effect that contains several filters such as bump mapping or edge enhancement that can be optimized. The optimization yields so-called parameter masks, which contain per-pixel parameter settings for each filter.

## 🚀 Try it out 🚀 
**Our demo is now on huggingface: [huggingface/Whitebox-Style-Transfer-Editing](https://huggingface.co/spaces/MaxReimann/Whitebox-Style-Transfer-Editing)**

![Streamlit Screenshot](images/screen_wise_demo.jpg?raw=true "WISE Editing Demo")

To run **locally**, clone the repo recursively and install the dependencies in requirements.txt. Set HUGGINGFACE to false in demo_config.py. 
Then run the streamlit app using `streamlit run Whitebox_style_transfer.py`



## Links & Paper 
[Project page](https://ivpg.hpi3d.de/wise/),
[arxiv link](https://arxiv.org/abs/2207.14606),
[framework code](https://github.com/winfried-ripken/wise)

"WISE: Whitebox Image Stylization by Example-based Learning", by Winfried Lötzsch*, Max Reimann*, Martin Büßemeyer, Amir Semmo, Jürgen Döllner, Matthias Trapp, in ECCV 2022

### Further notes
Pull Requests and further improvements welcome.
Please note that the shown effect is a minimal pipeline in terms of stylization capability, the much more feature-rich oilpaint and watercolor pipelines we show in our ECCV paper cannot be open-sourced due to IP reasons.

``` latex
@misc{loetzsch2022wise,
    title={WISE: Whitebox Image Stylization by Example-based Learning},
    author={Lötzsch, Winfried and Reimann, Max and Büssemeyer, Martin and Semmo, Amir and Döllner, Jürgen and Trapp, Matthias},
    year={2022},
    eprint={2207.14606},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
