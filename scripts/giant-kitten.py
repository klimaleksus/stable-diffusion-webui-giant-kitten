import code
import gc
import gradio as gr
import types
import torch
from modules import scripts,shared,script_callbacks,devices,sd_models

torch_cuda_is_available = torch.cuda.is_available()
torch_cuda_empty_cache = getattr(torch.cuda,'empty_cache') if torch_cuda_is_available else None
torch_cuda_ipc_collect = getattr(torch.cuda,'ipc_collect') if torch_cuda_is_available else None

GK_Persist = None
GK_Free = None

class GiantKitten(Exception):
    pass

def GK_Size(tensor):
    return tensor.numel()*tensor.element_size()

def GK_Noop(*ar,**kw):
    return None

def GK_Exit(*ar,**kw):
    if not torch_cuda_is_available:
        return
    global GK_Persist
    global GK_Free
    device = devices.device
    setattr(torch.cuda,'empty_cache',torch_cuda_empty_cache)
    setattr(torch.cuda,'ipc_collect',torch_cuda_ipc_collect)
    if GK_Persist is not None:
        GK_Persist.data = torch.tensor([])
        GK_Persist = None
    gc.collect()
    if GK_Free is not None:
        for tensor in GK_Free:
            tensor.data = tensor.data.to(device=device)
        GK_Free = None

def GK_Actions(actions,reserve,persist,noabort,reverse,ignored):
    if not torch_cuda_is_available:
        raise GiantKitten('FAIL! torch.cuda.is_available()==False')
    global GK_Persist
    global GK_Free
    device = devices.device
    GK_Exit()
    tensors = [obj for obj in gc.get_objects() if torch.is_tensor(obj) and (obj.device.type!='cpu') and (obj.device.type!='meta')]
    for tensor in tensors:
        tensor.data = tensor.data.cpu()
    if actions==1:
        GK_Free = tensors
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    else:
        active = (actions==2)
        unets = None
        if ignored:
            tensors.sort(key=GK_Size,reverse=reverse)
        else:
            ids = set([id(tensor) for tensor in tensors])
            del tensors
            if (shared.sd_model is None) or (shared.sd_model.model is None) or (shared.sd_model.model.diffusion_model is None):
                raise GiantKitten('FAIL! SD UNet model not found? To continue anyway, check "GK: ignored model"')
            shared.sd_model.model.diffusion_model.to(device=device)
            #sd_models.send_model_to_device(shared.sd_model)
            temp = [obj for obj in gc.get_objects() if torch.is_tensor(obj) and (obj.device.type!='meta')]
            temp.sort(key=GK_Size,reverse=reverse)
            unets = []
            tensors = []
            for tensor in temp:
                if tensor.device.type!='cpu':
                    unets.append(tensor)
                    tensor.data = tensor.data.cpu()
                elif id(tensor) in ids:
                    tensors.append(tensor)
            del temp
            del ids
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        kitten = None
        if active:
          if persist>0:
              GK_Persist = torch.ones((int(1024*1024*1024*persist),),dtype=torch.int8,device=device)
          if reserve>0:
              kitten = torch.ones((int(1024*1024*1024*reserve),),dtype=torch.int8,device=device)
        if unets is not None:
            for tensor in unets:
                tensor.data = tensor.data.to(device=device)
            del unets
        for tensor in tensors:
            tensor.data = tensor.data.to(device=device)
        del tensors
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if kitten is not None:
            kitten.data = torch.tensor([])
            del kitten
        if active:
            setattr(torch.cuda,'empty_cache',GK_Noop)
            setattr(torch.cuda,'ipc_collect',GK_Noop)
    if not noabort:
        raise GiantKitten('Done! (To keep generating images without deselecting this script, check "GK: no abort")')

script_callbacks.on_script_unloaded(GK_Exit)

class GK_Script(scripts.Script):
    def title(self):
        return 'Giant kitten (v1.0)'
    def ui(self,is_img2img):
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    gr.Markdown('### Giant kitten (v1.0)\n**Enable `CUDA - Sysmem Fallback Policy: Prefer Sysmem Fallback` in NVIDIA Control Panel!**')
                with gr.Column():
                    gr.Markdown("**Do not forget to un-select this script when you're done!**\n(But run with `GK: disable` once to reactivage Garbage Collector again)\n\nAlso, disable `--lowvram`/`--medvram` or anything similar.")
            with gr.Row():
                with gr.Tab('GK main action:'):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                gr_actions = gr.Radio(label='', choices=['GK: disable', 'GK: free VRAM', 'GK: activate'], value='GK: disable', type='index')
                                gr_noabort = gr.Checkbox(label='GK: no abort', value=False)
                            with gr.Row():
                                gr_reserve = gr.Slider(label='GK: reserve', minimum=0.0, maximum=24.0, step=0.1, value=0.0)
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown('- `disable` = free everything, sort to CUDA and allow GC (set this to undo all changes)\n- `free VRAM` = move everything to CPU (future generations will throw errors until disabled/activated later)\n- **`activate` = move to CPU, reserve, sort to CUDA, forbid GC and free reserved**\n- `no abort` = continue current generation even with this script enabled (to quickly test settings; will free VRAM and fill it again at each run)')
                    with gr.Row():
                        with gr.Group():
                            gr.Markdown('- **`reserve` = VRAM in GB that would be reserved in dedicated memory before moving everything to CUDA, but available for future allocations**\nTo find a good `reserve` value, run (aborting) with 0 and note your GPU usage. The amount of free dedicated memory is the minimal reserve you should approximately test.\nThen, note how much shared VRAM is used during a real generation (not aborting). You may further increase your reserve by this value, but try some intermediates too!')
                with gr.Tab('GK optional settings:'):
                    with gr.Row():
                        with gr.Column():
                            gr_reverse = gr.Checkbox(label='GK: sort reversed', value=False)
                        with gr.Column():
                            gr_ignored = gr.Checkbox(label='GK: ignored model', value=False)
                    with gr.Row():
                        gr_persist = gr.Slider(label='GK: persist', minimum=0.0, maximum=24.0, step=0.1, value=0.0)
                    with gr.Row():
                        gr.Markdown('- `sort reversed` = normally all tensors are sorted by size, from small to large; check this to sort down from large to small (so that big tensors would mostly stay in dedicated memory; may affect speed for good or bad depending on your `reserve` value\n- `ignored model` = normally tensors of SD UNet are moved to CUDA first; check this to treat all existing tensors equally\n- `persist` = allocate this much GB of VRAM and hold it while active (not recommended; keep at 0.0 unless testing as if you have GPU with less memory by this amount)\n\nMore info: https://github.com/klimaleksus/stable-diffusion-webui-giant-kitten')
        return [gr_actions,gr_reserve,gr_persist,gr_noabort,gr_reverse,gr_ignored]
    def run(self,p,gr_actions,gr_reserve,gr_persist,gr_noabort,gr_reverse,gr_ignored):
        GK_Actions(gr_actions,gr_reserve,gr_persist,gr_noabort,gr_reverse,gr_ignored)
#EOF
