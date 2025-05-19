# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .train import get_args_parser, main
from .ssl_meta_arch import SSLMetaArch
from .ssl_meta_arch_kd   import SSLMetaArch
from .ssl_meta_arch_LG2   import SSLMetaArch


from .train_s2 import get_args_parser, main
from .ssl_meta_arch_s2 import SSLMetaArch_s2
