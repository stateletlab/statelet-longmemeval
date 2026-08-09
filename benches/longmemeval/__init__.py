# Copyright 2024 Statelet Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LongMemEval Phase 0 benchmark harness for Statelet agent-memory.

Clean staged pipeline (Phase 0 of epic #741): inject all questions -> query all
questions -> hit@k -> DeepSeek answer + judge -> accuracy, mirroring mem0's
methodology (~94%).

See `run.py` for the single end-to-end command. The retrieval gate
(hit@k / recall@k) is LLM-free; only accuracy needs DEEPSEEK_API_KEY.
"""

from .abstention import (
    AbstentionPolicy,
    Confidence,
    calibrate,
    is_abstention,
)
from .config import CATEGORIES, MEM0_ACCURACY, DeepSeekConfig, HarnessConfig

__all__ = [
    "HarnessConfig",
    "DeepSeekConfig",
    "CATEGORIES",
    "MEM0_ACCURACY",
    "AbstentionPolicy",
    "Confidence",
    "calibrate",
    "is_abstention",
]
