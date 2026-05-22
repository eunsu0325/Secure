"""Loss functions for COCONUT"""

from .supcon import SupConLoss
from .proxy_anchor import ProxyAnchorLoss

__all__ = ['SupConLoss', 'ProxyAnchorLoss']