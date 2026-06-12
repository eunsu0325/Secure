"""Loss functions for COCONUT"""

from .proxy_anchor import ProxyAnchorLoss
from .ssl_consistency import SSLConsistencyLoss, VBMConsistencyLoss
from .softmax_head import SoftmaxHead

__all__ = ['ProxyAnchorLoss', 'SSLConsistencyLoss', 'VBMConsistencyLoss', 'SoftmaxHead']