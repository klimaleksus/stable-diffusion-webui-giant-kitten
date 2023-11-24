### Discussion: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/14077

# giant-kitten

This is Extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to use more efficiently the new System Fallback Policy of recent Nvidia driver on Windows.

![giant-kitten](https://klimaleksus2.ucoz.ru/sd/giant-kitten.jpg)

_You don't have to be a kitten to be giant._

## Installation:
Copy the link to this repository into `URL for extension's git repository` in WebUI `Extensions` → `Install from URL` tab:
```
https://github.com/klimaleksus/stable-diffusion-webui-giant-kitten
```
Also you may clone/download this repository and put it to `stable-diffusion-webui/extensions` directory.

Since this extension is actually a custom script, you can save [giant-kitten.py](https://github.com/klimaleksus/stable-diffusion-webui-giant-kitten/raw/master/scripts/giant-kitten.py) to `stable-diffusion-webui/scripts` instead.

## Requirements: 
- More than twice as much RAM as you have VRAM
- Windows 10+ with updated Nvidia drivers (version 546.01 and newer)
- Enabled `CUDA - System Fallback Policy` in "3D settings" of Nvidia Control Panel (either globally or at least for Python of WebUI venv) set to `Prefer System Fallback`

This extension is compatible with SD1/SDXL/ControlNet and whatever other stuff you might have in WebUI, including all other extensions.  
Though maybe it wouldn't give you any benefits.

## Background:
Previously, when pytorch wanted to move a tensor to CUDA memory, WebUI could throw OOM error if your `"dedicated"` GPU memory is already full.

Now, with "System Fallback" option enabled – pytorch will use `"shared"` GPU memory and move tensors there when dedicated memory is full. You won't see any errors until that shared memory will become full too.

### Caveats:
- Windows assigns half of your physical RAM to be available as "shared" memory if needed. For example, if you have 16 Gb of RAM and 6 Gb or VRAM, your "dedicated" GPU memory will be 6, and your shared GPU memory will be 8 (with "total" becoming 14 Gb)
- Shared memory is much slower than dedicated memory (approximately by 7-10 times), because any operation with a tensor there involves copying it from RAM to VRAM temporarily and sending results back. You will be capped with DRAM latency and PCIe bus speed.
- This offloading happens transparently in the video card driver. Pytorch cannot see "shared memory" as a separate device yet. Which means, python code cannot move a tensor from dedicated to shared or vice-versa, because it's all up to the driver.
- You can monitor your dedicated and shared GPU memory usage right in Windows Task Manager. For example, when you ask WebUI for a very large generation – you will see that after the "dedicated" graph reaches 100%, your "shared" graph starts to rise.

### Benefits:
- You can generate what you could not before! Your VRAM size can be increased by adding more RAM in your system. Yes, the speed will suffer, but at least you can try something that you couldn't earlier.
- No, "shared" GPU is not the same as "generating on CPU". Actually, tensor calculations on shared memory are much faster than the same calculations done on CPU, because it is still using CUDA kernels in half-precision mode (in reality, this is 10-20 times faster than pure CPU inference!)
- You can see exactly how much VRAM you will need to fit in dedicated VRAM. And when you do, the shared memory won't be used anymore, so there are no real drawbacks in leaving "Prefer System Fallback" enabled.
- Wait, actually there is one exceptional fault: if your generation is about to fill your dedicated memory up to its limit – then, most likely, the new Nvidia driver would move some tensors to shared area instead (probably to always have some free room in dedicated memory), thus reducing the overall speed. If you would disable the feature, setting it to `Prefer No System Fallback` in Nvidia Control Panel – then everything will have to fit solely in dedicated memory, and you'll get either maximal performance, or out-of-memory error.

## Idea:
1. Assume we found a way to move all tensors from CUDA to CPU before starting the generation. Now your GPU is completely empty, both dedicated and shared memory.
2. Now we will allocate a big tensor (several gigabytes in size) and "reserve" it.
3. Then we are returning the main Stable Diffusion model from CPU back to CUDA, along with everything that was needed for it.
4. Maybe some tensors didn't fit in dedicated memory (since a part of it was reserved) and have been offloaded to shared memory.
5. We free the reserved tensor, making its physical memory to be available for future allocations. This memory is guaranteed to be in "dedicated" area!
6. We start the generation. More tensors would be allocated as operational/temporary, but now pytorch will put them in the previously reserved area (instead of asking Windows to bring some fresh memory, which would have been from "shared" area otherwise).
7. Some of computations will use those operational tensors together with a part of model tensors that are still in dedicated memory, rather than always using operational tensors stored in shared memory.
8. In the best case, the total speed becomes faster by 20%-50% (relative to massive slowdown caused by shared memory) if the amount of reserved memory happened to be just what was needed to effectively store most important tensors.
9. We also should block pytorch Garbage Collector beforehand, so that it won't defragment or optimize anything after the generation completes – because otherwise the main model might be shifted back down to the dedicated area that was reserved previously.

This extension is a proof-of-concept that such optimization is possible indeed!

In attempt to get best of dedicated memory for most important tensors, they are moved back to CUDA in sorted order by their byte size from small to larger. But all tensors of the main SD model have priority, so your GPU memory would look like this:
1. Reserved gap (this will be freed after the allocation of everything else)
2. Smallest tensors of SD model UNet
3. Largest tensors of SD model UNet
4. All other active tensors, smallest
5. Largest of other tensors
6. Empty

If you check `GK: sort reversed` – all sorting will be from large to smaller. This may hurt or improve the overall performance. Though, preliminary tests show slowdowns when enabled.  
If you check `GK: ignored model` – the main SD model won't be detected separately, and all of available tensors would be sorted equally. It is expected to lower the speed if enabled.

# Usage:
## Preparation:
1. Install this extension. Close the WebUI and open Windows Task Manager. Make sure `System Fallback Policy` is set to `Prefer System Fallback` in Nvidia Control Panel.
2. Disable `medvram` or `lowvram` optimizations. You can leave sdp/xformers active, but no need in no-half/no-half-vae.
3. Run WebUI and load your preferred model (it can be SDXL if you want).
4. Set your VAE to TAESD in Settings od WebUI. Alternatively, you can use Tiled VAE from [MultiDiffusion Upscaler](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) if you know how.
5. Generate an image that you couldn't before (when you had an old driver, or when System Fallback was set to `Prefer No System Fallback`), or at least so large than you'll see "shared" memory graph rises after your "dedicated" graph becomes full.
6. Notice the speed in `s/it`, this is what should be improved by this extension. Also note the amount of shared memory that was used (the overflowed size).

If your GPU is already large enough (12 Gb and more), you won't overflow even with 2048x2048 resolution (with sdp/xformers but batch size = 1). Meaning that this extension is not for you, but you still can test it with artificially high resolution (edit your `ui-config.json`) or larger batch sizes, or with enabled `--no-half`  
Note that using lower batch sizes or enabling attention optimizations will always be faster than using shared memory otherwise.

## Activation:
This extension works as a custom script.

1. Go to txt2img or img2img tab. Choose custom script of this extension, it is called `Giant kitten (v1.0)`
2. While it is selected, choose the option `GK: activate` and run the generation as always.
3. It will be aborted by the script; no images would be created. But now all pytorch tensors that were on GPU have been moved to CPU and then back to CUDA in sorted order.
4. Check Windows Task Manager and note your current usage of dedicated GPU memory.
5. Return to WebUI and set the slider titled `GK: reserve` to some positive value that will be explained in the next section below.
6. Run the generation again, but it will be aborted as before. Now the memory layout has changed for all future generations (as long as you won't change the loaded SD model).
7. Deselect this script (choose `None` in custom script selection).
8. Finally, generate your actual image and notice its new generation speed!

(Instead of running the script twice and then deselecting it, you may check `GK: no abort` option of the script, so it will tweak your memory in the beginning of each generation without canceling it)

## How to calculate the correct "reserve" amount?

Reserved amount is the size of memory that will reside in dedicated area but will be empty when the generation starts, which means that the main SD model should be somewhere else: either in the rest of dedicated memory, or partially on shared memory.

- Reserve amount of 0 does nothing except for moving tensors back and forth, and thus should not give you any stable speed improvement (as long as bare SD model could fit into dedicated memory without overflowing).
- On the other hand, if your reserved amount will be equal (or more) than your actual dedicated GPU memory size – then the SD model is guaranteed to reside in shared memory, giving you the worst-case performance while still using dedicated area for operational calculations.
- Note the difference between your GPU size (dedicated) and current used amount of it, before you run the actual generation but after activating the script with zero reserve. For example, on 8 Gb video card, the loaded SDXL model takes 7.4/8, thus 0.6 Gb is free; while SD1 model takes 2.6/8, thus 5.4 Gb is free.
- This "free" dedicated memory is the approximate starting value for a proper reserve! You may try slightly less than this at first, for example 5 Gb for SD1 case above.
- Now, remember the "overflowed size" of your original generation, the amount of used shared memory? If we add it the previous value, we'll get the maximal reasonable reserve that won't increase your overall memory usage. For example, if in the above case during generation on SDXL there was 1.0 Gb used in shared GPU memory – then your approximate max reserve is 1.6 Gb (0.6 free-dedicated-before + 1.0 used-shared-after).
- Start from the minimal estimated reserve, and the increase it noticing the speed change. While testing, you don't have to perform a lot of steps, 2-4 would be enough to faithfully measure performance.
- Since the driver allocates memory in rather trickish way, with certain reserve you might sometimes get rid of any shared usage completely! This will be the maximal performance, but most likely it could have been achieved by disabling System Fallback completely for your specific case too.

## Additional options:

- Action `GK: free VRAM` will move all tensors to CPU without restoring them back to CUDA yet. This might be useful when you want to launch other GPU-intensive application; or to use WebUI for something else, for example when upscaling with LDSR. To continue as normal later, run with `GK: disable` or `GK: activate`
- Slider `GK: persist` (on the second tab of the script UI) will take the specified amount of VRAM before everything else, and won't free it. This can be used to simulate as if you had less GPU memory, but otherwise it should be zero.
- Checkboxes `GK: sort reversed` and `GK: ignored model` were already explained. You can try enabling their variations after you have found the best reserve value to see, whether it would improve the speed (most likely, it will not).

# F.A.Q

### TL;DR, how to use this!?

1. Enable `Prefer Fallback Policy` in NVIDIA, keep Task Manager opened
2. Pick `Giant kitten (v1.0)` as Script in WebUI, choose `GK: activate`
3. Set `GK: reserve` to something larger than zero but lower than your GPU size
4. Also check `GK: no abort` there, so it would be easier to test
5. Generate something extremely large and note the speed
6. Change `GK: reserve` and run again and again, measuring performance
7. When done, choose `GK: disable`, uncheck `GK: no abort`, run the last time and finally un-select the script

### Will this make my generations faster?

No, most likely not. This extension is a theoretical attempt to improve speed of shared memory usage. When pytorch will natively implement `tensor.shared()` and `tensor.dedicated()` as special versions of general `tensor.cuda()` – we will see much more effective optimizations!

### Can I use it on Linux or MacOS? What about AMD?

The code specifically relies on recent feature of Nvidia GPU driver. Without `System Fallback Policy` this extension cannot do anything useful. It will try anyway, as long as you have CUDA support in pytorch.

I don't know whether `Prefer System Fallback` can be enabled on Linux. If so, the script will work.  
AMD and MacOS are out of question, though.

### Some of sequential values of reserve are changing the speed drastically!

I suspect this has something to do with memory allocations. You see, python has its own object management, pytorch uses its own CUDA allocator, the operating system shares resources between all processes, and the video driver arranges byte arrays in VRAM or RAM.  
There are a lot of places where certain heuristics may lead to different memory layout.

Anyway, we shouldn't consider results as "stable" if just 100 Mb of allocation (0.1 Gb on the slider) is severely affecting the final speed.  
Try to find another reserved value for your particular case, which would not change so much!

### Why it is called a giant kitten?

![giant-kitten-meme](https://klimaleksus2.ucoz.ru/sd/giant-kitten-meme.jpg)
