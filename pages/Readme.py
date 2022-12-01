import streamlit as st

st.title("White-box Style Transfer Editing")

st.markdown("""
    This app demonstrates the editing capabilities of the White-box Style Transfer Editing (WISE) framework.
    It optimizes the parameters of classical image processing filters to match a given style image.
    After optimization, parameters can be tuned by hand to achieve a desired look.

    ### How does it work?
    We provide a small stylization effect that contains several filters such as bump mapping or edge enhancement that can be optimized. The optimization yields so-called parameter masks, which contain per pixel parameter settings of each filter.

    ### How to use the app ?
    - On the first page select existing content/style combinations or upload images to optimize. 
    - After the effect has been applied, use the parameter sliders to adjust a parameter value globally
    - On the "apply preset" page, we defined several parameter presets that can be drawn on the image. Press "Apply" to make the changes permanent
    - On the " local editing" page, individual parameter masks can be edited regionally. Choose the parameter on the left sidebar, and use the parameter strength slider to either increase or decrease the strength of the drawn strokes
    - Strokes on the drawing canvas (left column) are updated in real-time on the result in the right column. 
    - Strokes stay on the canvas unless manually deleted by clicking the trash button. To remove them from the canvas after each stroke, tick the corresponding checkbox in the sidebar.

    ### Links & Paper 
    [Project page](https://ivpg.hpi3d.de/wise/),
    [arxiv link](https://arxiv.org/abs/2207.14606)
    [demo code](https://github.com/MaxReimann/WISE-Editing)

    "WISE: Whitebox Image Stylization by Example-based Learning", by Winfried Lötzsch*, Max Reimann*, Martin Büßemeyer, Amir Semmo, Jürgen Döllner, Matthias Trapp, in ECCV 2022

    ### Further notes
    Pull Requests and further improvements are very welcome.
    On huggingface, optimization is currently disabled because it takes ~5min to optimize. If you want to try out your own images, please check out the app on [github](https://github.com/MaxReimann/WISE-Editing).
    Please note that the shown effect is a minimal pipeline in terms of stylization capability, the much more feature-rich oilpaint and watercolor pipelines we show in our ECCV paper cannot be open-sourced due to IP reasons.
""")