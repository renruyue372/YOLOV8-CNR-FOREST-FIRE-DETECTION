import math
import torch
import torch.nn as nn

__all__ = ["bbox_siou", "siou_loss", "SIoULoss"]

def _to_xyxy(box, xywh):
    if not xywh:
        return box
    x, y, w, h = box.unbind(-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)

def bbox_siou(box1, box2, xywh=True, eps=1e-7):
    b1 = _to_xyxy(box1, xywh)
    b2 = _to_xyxy(box2, xywh)

    b1_x1, b1_y1, b1_x2, b1_y2 = b1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = b2.unbind(-1)

    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h

    w1 = (b1_x2 - b1_x1).clamp(min=0)
    h1 = (b1_y2 - b1_y1).clamp(min=0)
    w2 = (b2_x2 - b2_x1).clamp(min=0)
    h2 = (b2_y2 - b2_y1).clamp(min=0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)).clamp(min=eps)
    ch = (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)).clamp(min=eps)

    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
    sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2).clamp(min=eps)

    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = math.sqrt(2.0) / 2.0

    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    sin_alpha = sin_alpha.clamp(0.0, 1.0)

    angle_cost = torch.cos(2.0 * torch.asin(sin_alpha) - math.pi / 2.0)

    rho_x = (s_cw / cw) ** 2
    rho_y = (s_ch / ch) ** 2
    gamma = angle_cost - 2.0
    distance_cost = 2.0 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    omega_w = torch.abs(w1 - w2) / (torch.max(w1, w2) + eps)
    omega_h = torch.abs(h1 - h2) / (torch.max(h1, h2) + eps)
    shape_cost = (1.0 - torch.exp(-omega_w)) ** 4 + (1.0 - torch.exp(-omega_h)) ** 4

    return iou - 0.5 * (distance_cost + shape_cost)

def siou_loss(pred, target, xywh=True, reduction="mean", eps=1e-7):
    loss = 1.0 - bbox_siou(pred, target, xywh=xywh, eps=eps)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction in ("none", None):
        return loss
    raise ValueError(f"Invalid reduction: {reduction!r}")

class SIoULoss(nn.Module):
    def __init__(self, xywh=True, reduction="mean", eps=1e-7):
        super().__init__()
        self.xywh = xywh
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        return siou_loss(pred, target, xywh=self.xywh, reduction=self.reduction, eps=self.eps)

if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.tensor([[50.0, 50.0, 100.0, 100.0]])
    b = torch.tensor([[50.0, 50.0, 100.0, 100.0]])
    print("identical :", bbox_siou(a, b).item())
    print("loss      :", siou_loss(a, b).item())
    c = torch.tensor([[500.0, 500.0, 10.0, 10.0]])
    print("disjoint  :", bbox_siou(a, c).item())
    p = torch.tensor([[10.0, 10.0, 20.0, 20.0],
                      [30.0, 40.0, 15.0, 25.0]], requires_grad=True)
    t = torch.tensor([[12.0, 11.0, 22.0, 18.0],
                      [35.0, 38.0, 16.0, 20.0]])
    loss = siou_loss(p, t)
    loss.backward()
    print("batch loss:", loss.item())
    print("grad ok   :", p.grad is not None and torch.isfinite(p.grad).all().item())
    p2 = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
    t2 = torch.tensor([[1.0, 1.0, 21.0, 19.0]])
    print("xyxy loss :", siou_loss(p2, t2, xywh=False, reduction="none").item())