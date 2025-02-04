# -*- coding: utf-8 -*-
from edgfs2D.post_process.base import BasePostProcessor  # noqa
from edgfs2D.post_process.compute_error import ComputeErrorPostProcessor  # noqa
from edgfs2D.post_process.vtu import VtuPostProcessor  # noqa
from edgfs2D.utils.nputil import subclass_where


def get_post_processor(name, *args, **kwargs):
    return subclass_where(BasePostProcessor, kind=name)(*args, **kwargs)
