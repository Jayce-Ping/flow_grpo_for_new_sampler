# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = 'sde',
    return_sqrt_dt: Optional[bool] = False,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    if sde_type == 'sde':
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level

        # our sde
        prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
            - torch.log(std_dev_t * torch.sqrt(-1*dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
    
    elif sde_type == 'cps':
        std_dev_t = sigma_prev  * math.sin(noise_level * math.pi / 2) # sigma_t in paper
        pred_original_sample = sample - sigma * model_output # predicted x_0 in paper
        noise_estimate = sample + model_output * (1 - sigma) # predicted x_1 in paper
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # remove all constants
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    elif sde_type == 'sde_dpm':
        # SDE-DPM-Solver++ order 1 (arXiv:2211.01095 Sec.5) on the flow schedule alpha = 1 - sigma,
        # generalized to a reverse-SDE diffusion coefficient scaled by lambda = noise_level, which
        # amounts to replacing the log-SNR step h by c = lambda**2 * h. lambda = 1 is the paper's
        # solver, lambda = 0 is the deterministic DPM-Solver++1 step (identical to Euler for RF).
        if noise_level < 0:
            raise ValueError(
                f"expected noise_level >= 0 for sde_type='sde_dpm', got {noise_level!r}"
            )
        alpha, alpha_prev = 1 - sigma, 1 - sigma_prev
        # exp(-h) is formed directly instead of via log-SNR so that sigma == 1 (alpha == 0) and
        # sigma_prev == 0 give 0 rather than log(0) = -inf.
        exp_neg_h = (alpha * sigma_prev) / (sigma * alpha_prev)
        if noise_level == 0:
            # Deterministic step, taken for every out-of-window step and for the whole eval
            # trajectory; set explicitly rather than leaning on 0**0 == 1 at the endpoints.
            decay = torch.ones_like(exp_neg_h)
        else:
            decay = exp_neg_h ** (noise_level ** 2)  # exp(-lambda**2 * h)
        std_dev_t = sigma_prev * torch.sqrt(1 - decay ** 2)
        # std is bounded by sigma_prev, so it collapses on the last steps of the schedule (sigma_prev
        # is 3e-3 on the second-to-last step of the 10-step SD3.5 grid) and 1/std**2 then blows the
        # ratio up. A legitimate window sits at std >= 0.25 on that grid, so 1e-2 separates the two.
        if noise_level > 0 and std_dev_t.min() < 1e-2:
            raise ValueError(
                f"sde_dpm std collapsed to {std_dev_t.min().item():.3e} at "
                f"sigma={sigma.flatten().tolist()} -> sigma_prev={sigma_prev.flatten().tolist()} "
                f"(noise_level={noise_level}); the log-prob would blow up as 1/std**2. "
                f"Keep the SDE window away from the final steps via sample.sde_window_range."
            )
        pred_original_sample = sample - sigma * model_output  # predicted x_0
        noise_estimate = sample + alpha * model_output        # predicted x_1
        # sqrt(sigma_prev**2 - std_dev_t**2) == sigma_prev * decay, so `decay` is used directly to
        # keep the coefficient exact and remove the negative-under-sqrt failure mode entirely.
        prev_sample_mean = pred_original_sample * alpha_prev + noise_estimate * sigma_prev * decay

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t**2)
            - torch.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    else:
        raise ValueError(
            f"expected sde_type in ('sde', 'cps', 'sde_dpm'), got {sde_type!r}"
        )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    if return_sqrt_dt:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1*dt)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t