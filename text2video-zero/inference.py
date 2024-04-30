import torch, os, argparse, json
import numpy as np
from model import Model


def read_prompts(prompt_path):
    prompts = []
    with open(prompt_path, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            data = json.loads(line)
            prompts.append(data["prompt"])
    return prompts


def inference_single_prompt(
    model,
    prompt,
    pid,
    output_root_path,
    with_watermark,
    fps=4,
    num_frame=8,
    model_path=None,
    seed=-1,
):
    print(model_path)
    if not with_watermark:
        out_file = f"{output_root_path}/{fps}fps_{num_frame}frames/{pid}_{prompt.replace(' ','_')}.mp4"
        out_path, _ = os.path.split(out_file)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print("--------------------------------------------")
        print(f"Generating video for {pid} prompt: {prompt} without watermark")
        print("--------------------------------------------")
        model.process_text2video(
            prompt,
            fps=fps,
            path=out_file,
            t0=44,
            t1=47,
            motion_field_strength_x=12,
            motion_field_strength_y=12,
            model_name=model_path,
            video_length=num_frame,
            seed=seed,
            watermark=None,
            with_watermark=with_watermark,
        )
    else:
        out_file = f"{output_root_path}/{fps}fps_{num_frame}frames/{pid}_{prompt.replace(' ','_')}_w.mp4"
        out_path, _ = os.path.split(out_file)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print("--------------------------------------------")
        print(f"Generating video for {pid} prompt: {prompt} with watermark")
        print("--------------------------------------------")
        model.process_text2video(
            prompt,
            fps=fps,
            path=out_file,
            t0=44,
            t1=47,
            motion_field_strength_x=12,
            motion_field_strength_y=12,
            model_name=model_path,
            video_length=num_frame,
            seed=seed,
            watermark=None,
            with_watermark=with_watermark,
        )


def inference_dataset(args):
    # prompts = ["A horse galloping on a street"]
    prompts = read_prompts(args.prompt_path)
    model = Model(device="cuda", dtype=torch.float16,reference_model=args.reference_model,reference_model_pretrain=args.reference_model_pretrain,)
    if not args.with_watermark:
        for pid, prompt in enumerate(prompts):
            inference_single_prompt(
                model=model,
                prompt=prompt,
                pid=pid + 586,
                output_root_path=args.output_root_path,
                fps=args.fps,
                num_frame=args.num_frame,
                model_path=args.model_path,
                seed=args.seed,
                with_watermark=False,
            )
    else:
        for pid, prompt in enumerate(prompts):
            inference_single_prompt(
                model=model,
                prompt=prompt,
                pid=pid + 796,
                output_root_path=args.output_root_path,
                fps=args.fps,
                num_frame=args.num_frame,
                model_path=args.model_path,
                seed=args.seed,
                with_watermark=True,
            )


if "__main__" == __name__:
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root_path", default="tmp_text_to_video_output_videos")
    parser.add_argument("--model_path", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--prompt_path", default="datas/fetv_data.json")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_frame", default=8, type=int)
    parser.add_argument("--fps", default=8, type=int)
    parser.add_argument("--with_watermark", default=True, type=bool)
    parser.add_argument("--reference_model", default="ViT-g-14")
    parser.add_argument("--reference_model_pretrain", default="laion2b_s12b_b42k")
    args = parser.parse_args()

    if args.seed is not None:
        _ = torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    inference_dataset(args)
