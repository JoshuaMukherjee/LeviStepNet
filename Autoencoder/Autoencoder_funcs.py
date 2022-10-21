import torch



def AE_vac(x):
    x = x.permute(0,2,1).contiguous()
    x = torch.view_as_complex(x)
    return x

def AE_var(x):
    x = torch.view_as_real(x)
    x = x.permute(0,2,1)
    return x