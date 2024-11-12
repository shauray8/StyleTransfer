import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import resize, normalize

def parse_args():
    parser = argparse.ArgumentParser(description="Add text overlay to an image and remove background.")
    parser.add_argument('image_path', type=str, help="Path to the input image.")
    parser.add_argument('text', type=str, help="Text to overlay on the image.")
    parser.add_argument('--x', type=int, default=125, help="X position for the text.")
    parser.add_argument('--y', type=int, default=500, help="Y position for the text.")
    parser.add_argument('--size', type=int, default=100, help="Font size for the text.")
    parser.add_argument('--color', type=str, default="(0, 0, 10, 200)", help="Text color in RGBA format.")
    
    return parser.parse_args()

def preprocess_image(im: np.ndarray, size: list) -> torch.Tensor:
    if im.shape[2] == 4: im = im[:, :, :3]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return normalize(im_tensor / 255.0, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

def postprocess_image(result: torch.Tensor, size: list) -> np.ndarray:
    result = torch.squeeze(resize(result, size), 0) 
    result = (result - result.min()) / (result.max() - result.min()) 
    result_image = (result * 255).byte().cpu().numpy() 
    if result_image.ndim == 3:
        result_image = result_image[0, :, :]  
    
    return result_image
def main():
    args = parse_args()

    orig_image = Image.open(args.image_path).convert("RGBA")
    orig_im_size = orig_image.size

    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_np = np.array(orig_image)
    model_input_size = [512, 512]
    image_tensor = preprocess_image(image_np, model_input_size).to(device)

    with torch.no_grad():
        result = model(image_tensor)

    result_image = postprocess_image(result[0][0], orig_im_size)

    subject_mask = Image.fromarray(result_image).convert("L").resize(orig_image.size)
    subject_image = Image.new("RGBA", orig_image.size)
    subject_image.paste(orig_image, mask=subject_mask)

    # Parse color
    text_color = tuple(map(int, args.color.strip("()").split(",")))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", args.size)

    text_image = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_image)
    bbox = draw.textbbox((0, 0), args.text, font=font)
    draw.text((args.x, args.y), args.text, font=font, fill=text_color)

    final_image = Image.alpha_composite(orig_image, text_image)
    final_image = Image.alpha_composite(final_image, subject_image)

    final_image.save("output.png")

if __name__ == "__main__":
    main()

