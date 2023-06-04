import cv2
import argparse
import gradio as gr
import numpy as np

# MMOCR
from mmocr.apis.inferencers import MMOCRInferencer

# BUILD MMOCR


def arg_parse():
    parser = argparse.ArgumentParser(description='MMOCR demo for gradio app')
    parser.add_argument(
        '--rec_config',
        type=str,
        default='mmocr-dev-1.x/configs/textrecog/maerec/maerec_b_union14m.py',
        help='The recognition config file.')
    parser.add_argument(
        '--rec_weight',
        type=str,
        default=
        'mmocr-dev-1.x/work_dirs/maerec_b_union14m/maerec_b_union14m.pth',
        help='The recognition weight file.')
    parser.add_argument(
        '--det_config',
        type=str,
        default=
        'mmocr-dev-1.x/configs/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py',  # noqa,
        help='The detection config file.')
    parser.add_argument(
        '--det_weight',
        type=str,
        default='mmocr-dev-1.x/dbnetpp.pth',
        help='The detection weight file.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='The device used for inference.')
    args = parser.parse_args()
    return args


def run_mmocr(img: np.ndarray, use_detector: bool = True):
    """Run MMOCR and SAM

    Args:
        img (np.ndarray): Input image
        use_detector (bool, optional): Whether to use detector. Defaults to
            True.
    """
    if use_detector:
        mode = 'det_rec'
    else:
        mode = 'rec'
    # Build MMOCR
    mmocr_inferencer.mode = mode
    result = mmocr_inferencer(img, return_vis=True)
    visualization = result['visualization'][0]
    result = result['predictions'][0]

    if mode == 'det_rec':
        rec_texts = result['rec_texts']
        det_polygons = result['det_polygons']
        det_results = []
        for rec_text, det_polygon in zip(rec_texts, det_polygons):
            det_polygon = np.array(det_polygon).astype(np.int32).tolist()
            det_results.append(f'{rec_text}: {det_polygon}')
        out_results = '\n'.join(det_results)
        visualization = cv2.cvtColor(
            np.array(visualization), cv2.COLOR_RGB2BGR)
    else:
        rec_text = result['rec_texts'][0]
        rec_score = result['rec_scores'][0]
        out_results = f'pred: {rec_text} \n score: {rec_score:.2f}'
        visualization = None
    return visualization, out_results


if __name__ == '__main__':
    args = arg_parse()
    mmocr_inferencer = MMOCRInferencer(
        args.det_config,
        args.det_weight,
        args.rec_config,
        args.rec_weight,
        device=args.device)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                    <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
                    <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                        MAERec: A MAE-pretrained Scene Text Recognizer
                    </h1>
                    <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem"> 
                    [<a href="https://arxiv.org/abs/2305.10855" style="color:blue;">arXiv</a>] 
                    [<a href="https://github.com/Mountchicken/Union14M" style="color:green;">Code</a>]
                    </h3>
                    <h2 style="text-align: left; font-weight: 600; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
                    MAERec is a scene text recognition model composed of a ViT backbone and a Transformer decoder in auto-regressive
                    style. It shows an outstanding performance in scene text recognition, especially when pre-trained on the
                    Union14M-U through MAE.
                    </h2>
                    <h2 style="text-align: left; font-weight: 600; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
                    In this demo, we combine MAERec with DBNet++ to build an
                    end-to-end scene text recognition model.
                    </h2>
                    </div>
                    """)
                gr.Image('github/maerec.png')
            with gr.Column(scale=1):
                input_image = gr.Image(label='Input Image')
                output_image = gr.Image(label='Output Image')
                use_detector = gr.Checkbox(
                    label=
                    'Use Scene Text Detector or Not (Disabled for Recognition Only)',
                    default=True)
                det_results = gr.Textbox(label='Detection Results')
                mmocr = gr.Button('Run MMOCR')
                gr.Markdown("## Image Examples")
        with gr.Row():
            gr.Examples(
                examples=[
                    'github/author.jpg', 'github/gradio1.jpeg',
                    'github/Art_Curve_178.jpg', 'github/cute_3.jpg',
                    'github/cute_168.jpg', 'github/hiercurve_2229.jpg',
                    'github/ic15_52.jpg', 'github/ic15_698.jpg',
                    'github/Art_Curve_352.jpg'
                ],
                inputs=input_image,
            )
        mmocr.click(
            fn=run_mmocr,
            inputs=[input_image, use_detector],
            outputs=[output_image, det_results])
    demo.launch(debug=True)
