"""Loss functions for COCONUT"""

from .supcon import SupConLoss
from .proxy_anchor import ProxyAnchorLoss
from .idl_rtm import IDL_RTMLoss

__all__ = ['SupConLoss', 'ProxyAnchorLoss', 'IDL_RTMLoss']