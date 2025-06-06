## Muon code from Moonlight
## https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
import torch
import math
import warnings

@torch.compile  ## I had to comment this out, it was throwing an error
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X



class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        lmo: Whether to use LMO instead variational viewpoint of gradient descent to derive
        update rule. If lmo=False, update is additionally scaled by the dual norm of the
        gradient.
        l2_prod_norm: Whether to use the L2 norm for the product space over layers
        instead of the max norm, which scales each layer's LR by the nuclear norm of the
        gradient.
        rms_layer_norm: Whether to use the RMS norm the input/output space of each
        layer, which scale each layer's LR by sqrt(fan_out/fan_in).
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self,
                 named_params,
                 lr=1e-3,
                 wd=0.1,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 lmo=True,
                 l2_prod_norm=False,
                 rms_layer_norm=False,
                 adamw_betas=(0.95, 0.95),
                 adamw_eps=1e-8):
            
        defaults = dict(
                lr=lr,
                wd=wd,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                lmo=lmo,
                l2_prod_norm=l2_prod_norm,
                rms_layer_norm=rms_layer_norm,
                adamw_betas=adamw_betas,
                adamw_eps=adamw_eps,
        )
        
        muon_params, muon_params_names = [], []
        adamw_params, adamw_params_names = [], []
        for name, p in named_params:
            if p.ndim >= 2 and not any(excluded in name for excluded in ["embeddings", "embed_tokens", "wte", "lm_head", "wpe"]):
                muon_params.append(p)
                muon_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)
        # print("EMBED TOKENS AND LM_HEAD SHOULD BE WITH ADAMW.")
        # print("Params trained with MUON : ", muon_params_names)
        # print("Params trained with ADAMW : ", adamw_params_names)
        params = list(muon_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
        # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
        for p in muon_params:
                assert p.ndim == 2, p.ndim
                self.state[p]["use_muon"] = True
                
        for p in adamw_params:
                # Do not use Muon for parameters in adamw_params
                self.state[p]["use_muon"] = False

    def step(self, closure=None):
        """Perform a single optimization step.
            Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        loss = None
        if closure is not None:
                with torch.enable_grad():
                        loss = closure()
                        
        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            lmo = group["lmo"]
            l2_prod_norm = group["l2_prod_norm"]
            rms_layer_norm = group["rms_layer_norm"]

            # initial pass over parameters to compute update direction and LR scalings.
            # Warning for the future: if we ever use more than one param group, these
            # scalings are not going to behave exactly right. Here we compute scaling
            # factors that depend on all layers of the network, so we assume that all
            # layers of the network are inside the current param group.
            layer_nuc_norms = None
            for i, p in enumerate(params):

                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # calc momentum.
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                # quit now if update doesn't depend on nuclear norm of layer gradients.
                if lmo and not l2_prod_norm:
                    continue

                # calc update.
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # compute nuclear norms for each layer. Gotta do some case analysis
                # here depending on combination of options for l2 vs. max product norm
                # and lmo vs. variational veiewpoint.
                # should we cast g to 16-bit here or can we just do it once earlier?
                if layer_nuc_norms is None:
                    layer_nuc_norms = torch.zeros(len(params), device=p.device)
                layer_nuc_norms[i] = torch.trace(g.bfloat16().T @ u)
                if rms_layer_norm:
                    fan_out, fan_in = p.shape[:2]
                    layer_nuc_norms[i] *= math.sqrt(fan_out / fan_in)

            # compute lr scaling factors that depend on all layers. doing this here so
            # we don't recompute this for every layer unnecessarily.
            if lmo and l2_prod_norm:
                global_dual_norm = torch.linalg.vector_norm(layer_nuc_norms, ord=2)
            if not lmo and not l2_prod_norm:
                global_dual_norm = torch.sum(layer_nuc_norms)

            # apply weight updates
            for i, p in enumerate(params):

                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # calc update. Note that we already computed and stored the momentum
                # term before, but we are re-computing the matrix sign. This is
                # suboptimal w.r.t.  time but doesn't use any extra memory. We can
                # always tweak this later.
                buf = self.state[p]["momentum_buffer"]
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # apply scaling factors to lr depending on steepest descent variations
                lr_scale = 1.0
                if lmo and not l2_prod_norm:
                    if rms_layer_norm:
                        fan_out, fan_in = p.shape[:2]
                        lr_scale = math.sqrt(fan_out / fan_in)
                if lmo and l2_prod_norm:
                    lr_scale = layer_nuc_norms[i] / global_dual_norm
                if not lmo and not l2_prod_norm:
                    lr_scale = global_dual_norm
                if not lmo and l2_prod_norm:
                    lr_scale = layer_nuc_norms[i]
                adjusted_lr = lr_scale * lr

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)
                
            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)
                    
        return loss



