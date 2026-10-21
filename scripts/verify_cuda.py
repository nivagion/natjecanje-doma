"""Fail-closed real CUDA forward and backward verification."""
import json
import subprocess
import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device=torch.device("cuda:0")
    x=torch.randn(1024,1024,device=device,requires_grad=True)
    w=torch.nn.Linear(1024,128).to(device)
    y=w(x).square().mean();y.backward();torch.cuda.synchronize()
    free,total=torch.cuda.mem_get_info()
    out={"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
         "capability":torch.cuda.get_device_capability(0),"bf16_torch_emulated_or_native":torch.cuda.is_bf16_supported(),
         "bf16_native":torch.cuda.get_device_capability(0)[0]>=8 and torch.cuda.is_bf16_supported(),
         "tensor_device":str(x.device),"parameter_device":str(next(w.parameters()).device),
         "gradient_ok":bool(torch.isfinite(w.weight.grad).all()),"allocated_mb":round(torch.cuda.memory_allocated()/1048576),
         "free_mb":round(free/1048576),"total_mb":round(total/1048576)}
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
