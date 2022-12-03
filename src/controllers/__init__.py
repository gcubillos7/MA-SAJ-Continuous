REGISTRY = {}

from .basic_controller import BasicMAC
from .rode_controller import RODEMAC
from .role_controller import ROLEMAC
from .continous_controller import CMAC
REGISTRY["basic_mac"] = BasicMAC
REGISTRY['rode_mac'] = RODEMAC
REGISTRY['role_mac'] = ROLEMAC
REGISTRY['continous_mac'] = CMAC