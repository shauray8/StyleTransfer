# Diffusion Style Transfer with IP-Adapter and Deadiff

Experimenting with combining DEADiff and IP-Adapters in order to get a pipeline that transfers style keeping the Image consistent through the pipeline, let's see how this goes!

## What’s Happening Here?
working with diffusion models, leveraging the power of IP-Adapter and Deadiff to pull off some magic for style transfer. Here’s how:
* IP-Adapter: Involves adding an image encoder that merges its output via cross-attention with the text encoder's output. The merged data is fed into every block of the U-Net. Using this to keep facial data consistent and realistic.

* Deadiff: This one brings a bit more structure to the table. It passes the reference image through two blocks of a Q-Former – one for style and one for schematics. The results are then piped into the U-Net, adding layers of control over style and structure.

## Why Combine?
I’m blending the best of both worlds—IP-Adapter for maintaining facial features and Deadiff for more control over style and structure. The result is a more refined style transfer model that ensures the face looks right while letting you play around with the style. In theory atleast :)

## TL;DR
1. IP-Adapter keeps facial features on point using cross-attention and U-Nets.
2. Deadiff adds more control, using Q-Former for detailed style and structure manipulation.
3. Together, they make style transfer cleaner and more controlled.

feel free to contribute!
<br>

--- 
1. https://arxiv.org/abs/2403.06951
2. https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter

