import torch
import math

def make_linear_beta_schedule(
        num_steps=50,
        beta_start=1e-4,
        beta_end=0.02,
        device=None,
):
    return torch.linspace(
        beta_start,
        beta_end,
        num_steps,
        dtype=torch.float32,
        device=device,
    )


def make_cosine_beta_schedule(
        num_steps=50,
        s=0.008,
        device=None,
):
    steps = num_steps + 1

    x = torch.linspace(
        0,
        num_steps,
        steps,
        dtype=torch.float32,
        device=device,
    )

    alpha_bar = torch.cos(
        (
            (x / num_steps + s)
            / (1.0 + s)
        )
        * math.pi
        / 2.0
    ) ** 2

    alpha_bar = (
        alpha_bar
        / alpha_bar[0]
    )

    beta = 1.0 - (
        alpha_bar[1:]
        / alpha_bar[:-1]
    )

    return torch.clamp(
        beta,
        min=1e-5,
        max=0.999,
    )
class DiffusionSchedule:
    def __init__(
            self,
            num_steps=50,
            schedule_type="cosine",
            beta_start=1e-4,
            beta_end=0.02,
            device="cuda",
    ):
        self.num_steps = num_steps

        if schedule_type == "cosine":
            self.beta = make_cosine_beta_schedule(
                num_steps=num_steps,
                device=device,
            )
        elif schedule_type == "linear":
            self.beta = make_linear_beta_schedule(
                num_steps=num_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown schedule_type: "
                f"{schedule_type}"
            )

        self.alpha = 1.0 - self.beta

        self.alpha_bar = torch.cumprod(
            self.alpha,
            dim=0,
        )

        self.sqrt_alpha_bar = torch.sqrt(
            self.alpha_bar
        )

        self.sqrt_one_minus_alpha_bar = torch.sqrt(
            1.0 - self.alpha_bar
        )