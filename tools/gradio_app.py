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
        default=  # noqa
        '/media/jiangqing/jqssd/projects/research/DGDataset/Union14M/mmocr-dev-1.x/work_dirs/maerec_b_union14m/maerec_b_union14m.pth',  # noqa
        help='The recognition weight file.')
    parser.add_argument(
        '--det_config',
        type=str,
        default=  # noqa
        'mmocr-dev-1.x/configs/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py',  # noqa
        help='The detection config file.')
    parser.add_argument(
        '--det_weight',
        type=str,
        default=  # noqa
        '/media/jiangqing/jqssd/projects/research/DGDataset/Union14M/mmocr-dev-1.x/dbnetpp.pth',  # noqa
        help='The detection weight file.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='The device used for inference.')
    args = parser.parse_args()
    return args


def run_mmocr(img: np.ndarray, inferencer: MMOCRInferencer):
    """Run MMOCR and SAM

    Args:
        img (np.ndarray): Input image
        inferencer (MMOCRInferencer): MMOCR Inferencer.
    """
    # Build MMOCR

    result = mmocr_inferencer(img, return_vis=True)
    visualization = result['visualization'][0]
    result = result['predictions'][0]
    rec_texts = result['rec_texts']
    det_polygons = result['det_polygons']
    det_results = []
    for rec_text, det_polygon in zip(rec_texts, det_polygons):
        det_results.append(f'{rec_text}: {det_polygon}')
    det_results = '\n'.join(det_results)
    visualization = cv2.cvtColor(np.array(visualization), cv2.COLOR_RGB2BGR)
    return visualization, det_results


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
                input_image = gr.Image(label='Input Image')
                det_results = gr.Textbox(label='Detection Results')
                mmocr = gr.Button('Run End2End MMOCR')

            with gr.Column(scale=1):
                output_image = gr.Image(label='Output Image')
                gr.Markdown("## Image Examples")
                gr.Examples(
                    examples=[
                        'github/gradio1.jpeg', 'github/gradio2.png'
                    ],
                    inputs=input_image,
                )
            mmocr.click(
                fn=run_mmocr,
                inputs=[input_image],
                outputs=[output_image, det_results])
    demo.launch(debug=True)
