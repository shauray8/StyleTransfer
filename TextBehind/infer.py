import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Add text overlay to an image and remove background.")
    parser.add_argument('image_path', type=str, help="Path to the input image.")
    parser.add_argument('text', type=str, help="Text to overlay on the image.")
    parser.add_argument('--x', type=int, default=125, help="X position for the text.")
    parser.add_argument('--y', type=int, default=500, help="Y position for the text.")
    parser.add_argument('--size', type=int, default=100, help="Font size for the text.")
    parser.add_argument('--color', type=str, default="(0, 0, 10, 200)", help="Text color in RGBA format.")
    return parser.parse_args()

# Background removal function
def remove_background(image, model, device):
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_rgb = image.convert("RGB")
    input_image = transform_image(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_image)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    mask = transforms.ToPILImage()(pred).resize(image.size)

    # Create an RGBA image with alpha channel based on the mask
    image_no_bg = image.copy()
    image_no_bg.putalpha(mask)
    return image_no_bg

def main():
    args = parse_args()

    orig_image = Image.open(args.image_path).convert("RGBA")
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    subject_image = remove_background(orig_image, model, device)

    background_with_text = orig_image.copy()
    background_with_text.paste(subject_image, mask=subject_image.split()[-1])  # Ensures original background with subject mask

    text_color = tuple(map(int, args.color.strip("()").split(",")))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", args.size)
    draw = ImageDraw.Draw(background_with_text)
    draw.text((args.x, args.y), args.text, font=font, fill=text_color)

    final_image = Image.alpha_composite(background_with_text, subject_image)

    final_image.save("output.png")

if __name__ == "__main__":
    main()

