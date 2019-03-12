from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss
from .kl_div_loss import KLDivLoss
from .mean_squared_error_loss import SelectedMSELoss
from .hard_mine_triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ring_loss import RingLoss


def DeepSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
