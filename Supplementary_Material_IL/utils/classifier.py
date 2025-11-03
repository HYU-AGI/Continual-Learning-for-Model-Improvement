import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from utils.backbone import get_backbone, obtain_features

def get_classifier(params, feature_dim, out_dim_list):
    if params.classifier == 'None':
        return None
    elif params.classifier == 'CosineLinear':
        return nn.ModuleList([CosineLinear(in_features=feature_dim,out_features=out_dim_list[t_id]) for t_id in range(len(out_dim_list))])
    elif params.classifier == 'Linear':
        return nn.ModuleList([nn.Linear(in_features=feature_dim,out_features=out_dim_list[t_id]) for t_id in range(len(out_dim_list))])
    elif params.classifier == 'MaskedCosineLinear':          
    # CosineLinear를 먼저 만든 뒤 build_classifier에서 래핑
        return nn.ModuleList([CosineLinear(feature_dim, out_dim_list[t]) for t in range(len(out_dim_list))])
    else:
        raise NotImplementedError()
    
def get_real_weight(layer):
    """
    WeightMaskWrapper인지 여부에 관계없이 weight tensor를 반환
    """
    if isinstance(layer, WeightMaskWrapper):
        return layer.base_layer.weight
    return layer.weight

# utils/classifier.py  ── 기존 compute_importance 함수 교체
def compute_importance(layer, loader, *, model, tokenizer, params,
                       metric="fisher", device=None):

    device  = device or next(model.parameters()).device
    weight  = get_real_weight(layer)
    imp_acc = torch.zeros_like(weight)

    layer = layer.to(device)
    model.eval(); layer.eval()

    for lm_input in loader:
        # ---------- 1) feature 추출(백본) : grad 필요 없음 ----------
        with torch.no_grad():
            feats = obtain_features(params=params,
                                    model=model,
                                    lm_input=lm_input,
                                    tokenizer=tokenizer).to(device)

        # ---------- 2) Classifier forward : grad 필요 ----------
        if metric == "fisher":
            logits = layer(feats)
            prob   = torch.softmax(logits, dim=-1)

            for i in range(prob.size(0)):
                layer.zero_grad(set_to_none=True)
                prob[i].log().sum().backward(retain_graph=True)
                imp_acc += weight.grad.abs()**2      # (∂log p)^2

        elif metric == "mas":
            layer.zero_grad(set_to_none=True)
            out = layer(feats)
            (out**2).sum().backward()
            imp_acc += weight.grad.abs()            # |∂ ‖F‖²|

        else:
            raise ValueError("metric must be 'fisher' or 'mas'")

    imp_acc /= len(loader.dataset)
    return imp_acc.detach()
    
class MultiProtoCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, num_proto=5):
        super(MultiProtoCosineLinear, self).__init__()
        self.num_proto = num_proto
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features*num_proto, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        # out = out.view(-1, self.out_features, self.num_proto).max(dim=-1).values
        return out
        

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))

        return out

# ===== 추가 모듈: 기존 Linear/CosineLinear를 감싸는 래퍼 =====
class WeightMaskWrapper(nn.Module):
    """
    아무 Linear·CosineLinear 계층도 감싸서
    - mask 값이 1인 weight만 학습
    - mask 값이 0인 weight은 gradient를 0으로 만듦
    """
    def __init__(self, base_layer: nn.Module, init_mask: torch.Tensor):
        super().__init__()
        self.base_layer = base_layer
        self.register_buffer("mask", init_mask.clone().float())

        # backward 때 gradient 마스킹
        self.base_layer.weight.register_hook(lambda g: g * self.mask)

    # mask를 바꿔 달라고 할 때 호출
    @torch.no_grad()
    def update_mask(self, new_mask: torch.Tensor):
        assert new_mask.shape == self.mask.shape
        self.mask.copy_(new_mask.float())

    def forward(self, x):
        w = self.base_layer.weight                  # 실제 학습될 파라미터
        w_masked = w * self.mask                   # 값만 0/1로 가린다

        # ── CosineLinear인 경우 동일 수식 재작성 ──
        if isinstance(self.base_layer, CosineLinear):
            out = F.linear(F.normalize(x, dim=1),
                           F.normalize(w_masked, dim=1))
        else:   # Linear 등
            out = F.linear(x, w_masked, self.base_layer.bias)

        # grad 마스킹은 이미 register_hook으로 처리 완료
        return out