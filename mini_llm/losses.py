import torch
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor


def ce_loss(output: Tensor, target: LongTensor) -> torch.float:
    r"""Cross entropy loss.

    b: batch_size
    l: seq_len
    v: vocab_size

    Args:
        output: (b, l, v)
        target: (b, l)

    Outputs:
        loss: torch.float
    """

    loss = F.cross_entropy(
        input=rearrange(output, 'b l v -> (b l) v'),
        target=rearrange(target, 'b l -> (b l)'),
        ignore_index=-1
    )

    return loss