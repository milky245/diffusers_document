from diffusers import DDPMPipeline
from diffusers import DDPMScheduler, UNet2DModel
import torch
import numpy as np
from PIL import Image

# ddpm=DDPMPipeline.from_pretrained("google/ddpm-cat-256",use_safetensors=True)
# image=ddpm(num_inference_steps=50).images[0]
# image.show()

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model=UNet2DModel.from_pretrained("google/ddpm-cat-256",use_safetensors=True)
scheduler.set_timesteps(50)
print(scheduler.timesteps)

sample_size=model.config.sample_size
noise=torch.randn((1,3,sample_size,sample_size))

#write a loop to iterate over the timesteps
input=noise
# This is the entire denoising process, 
# and you can use this same pattern to write any diffusion system.
for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual=model(input, t).sample
    previous_noisy_sample=scheduler.step(noisy_residual,t,input).prev_sample
    input=previous_noisy_sample

# convert the denoised output to an image
image=(input/2+0.5).clamp(0,1).squeeze()
image=(image.permute(1,2,0)*255).round().to(torch.uint8).cpu().numpy()
image=Image.fromarray(image)
image.show()

