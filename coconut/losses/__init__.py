"""Loss functions for COCONUT"""

from .proxy_anchor import ProxyAnchorLoss
from .ssl_consistency import SSLConsistencyLoss, VBMConsistencyLoss

__all__ = ['ProxyAnchorLoss', 'SSLConsistencyLoss', 'VBMConsistencyLoss']