# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

import copy
import logging
import pdb
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch

from ..cache_utils import DynamicCache
from ..tokenization_utils import PreTrainedTokenizer
from .logits_process import LogitsProcessorList, MinLengthLogitsProcessor


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..tokenization_utils_fast import PreTrainedTokenizerFast
    from .configuration_utils import GenerationConfig


def convert_token_ids(
    input_ids: torch.Tensor,
    src: Union[PreTrainedTokenizer, "PreTrainedTokenizerFast"],
    dest: Union[PreTrainedTokenizer, "PreTrainedTokenizerFast"],
):
    text = src.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    dest_ids = dest(text, add_special_tokens=True, return_tensors="pt")["input_ids"]
    return dest_ids.to(input_ids.device)


# def get_sub_seq(a, b):
#     for i in range(a.shape[1] - b.shape[1]):
#         if (a[:, i : i + b.shape[1]] == b).all():
#             return i
#     return None


# def get_only_new_tokens(a, b):
#     c = get_sub_seq(a, b)
#     return a[:, c + b.shape[1] :]

def get_sub_seq(a, b):
    i_agree_max = None
    agree_max = 0
    for i in range(a.shape[1] - b.shape[1]):
        agree_seq = a[:, i : i + b.shape[1]] == b
        agree = agree_seq.sum()
        if agree_max < agree:
            agree_max = agree
            i_agree_max = i
    return i_agree_max

def get_only_new_tokens(a, b):
    c = get_sub_seq(a, b)
    return a[:, c + b.shape[1] :]

class CandidateGenerator:
    """Abstract base class for all candidate generators that can be applied during assisted generation."""

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and, optionally, a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `get_candidates`."
        )

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call "
            "`update_candidate_strategy`."
        )


class AssistedCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for assisted generation and speculative decoding. This class generates
    candidates through the use of a smaller model. Read the following blog post for more information:
    https://huggingface.co/blog/assisted-generation

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        assistant_model (`PreTrainedModel`):
            The model to be used for generating candidates. This model should be smaller than the main model.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        model_kwargs (`Dict`):
            The keyword arguments that will be passed to the main model, and are used as base inputs for the assistant
            model as well.
        inputs_tensor (`torch.Tensor`, *optional*):
            The model input tensor. In encoder-decoder models, this is the encoder input.
    """

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        logits_processor: "LogitsProcessorList" = None,
        assistant_tokenizer: Optional[Union[PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
        target_tokenizer: Optional[Union[PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    ):
        device = assistant_model.device

        if assistant_tokenizer != target_tokenizer:
            input_ids = convert_token_ids(input_ids, src=target_tokenizer, dest=assistant_tokenizer)
            self.different_tokenizers = True
            self.assistant_tokenizer = assistant_tokenizer
            self.target_tokenizer = target_tokenizer

        inputs_tensor = input_ids.to(device)

        # Make sure all data at the same device as assistant model
        input_ids = input_ids.to(device)
        if inputs_tensor is not None:
            inputs_tensor = inputs_tensor.to(device)

        # Prepare the assistant and the starting number of candidate tokens
        self.assistant_model = assistant_model
        self.num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens

        # Prepare the kwargs for the assistant model
        assistant_kwargs = {}
        for key, value in model_kwargs.items():  # deepcopy crashes if we attempt to copy encoder outputs with grads
            if key not in ("encoder_outputs", "assistant_encoder_outputs", "past_key_values"):
                assistant_kwargs[key] = (
                    value.detach().to(device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                )

        if "assistant_encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["assistant_encoder_outputs"]
        elif assistant_model.config.is_encoder_decoder:
            inputs_tensor, model_input_name, assistant_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_kwargs
            )
            assistant_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_kwargs, model_input_name, assistant_model.generation_config
            )
        elif "encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs"]
        self.assistant_kwargs = assistant_kwargs

        # Prepare assistant model's keys of inputs
        if assistant_model.config.is_encoder_decoder:
            # both are encoder-decoder
            self.input_ids_key = "decoder_input_ids"
        elif "encoder_outputs" in assistant_kwargs:
            # special case for encoder-decoder with decoder-only assistant (like DistilWhisper)
            self.input_ids_key = "input_ids"
            self.assistant_kwargs["attention_mask"] = self.assistant_kwargs.get(
                "decoder_attention_mask",
                torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long),
            )
        else:
            # both are decoder-only
            self.input_ids_key = "input_ids"

        # Prepare generation-related options.
        self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        self.generation_config = copy.deepcopy(generation_config)
        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True

        # Disable sampling -- this implementation of assisted generation/speculative decoding uses the assistant
        # greedily to maximize matches. Disables sampling-related flags to prevent warnings
        self.generation_config.do_sample = False
        for attr in ("temperature", "top_p", "min_p", "typical_p", "top_k", "epsilon_cutoff", "eta_cutoff"):
            setattr(self.generation_config, attr, None)

        # avoid unnecessary warnings that min_length is larger than max_new_tokens
        # remove the `MinLengthLogitsProcessor` if exists (NOTE: no need to check for `MinNewTokensLogitsProcessor`)
        self.main_model_min_length = self.generation_config.min_length
        self.generation_config.min_length = 0
        self.generation_config.min_new_tokens = None
        for processor in self.logits_processor:
            if type(processor) == MinLengthLogitsProcessor:
                raise ValueError(
                    "Passing `MinLengthLogitsProcessor` when using `assisted_generation is disabled. "
                    "Please pass in `min_length` into `.generate()` instead"
                )
        self.prev_tokens = None
        self.target_lookbehind = 10
        self.draft_lookbehind = 10

    def get_candidates(self, input_ids: torch.LongTensor, stopping_criteria=None) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        # some = torch.tensor([[2, 38182, 3916, 2072, 35, 50118, 1640, 16256, 43, 38, 192, 2434, 9, 10, 7977, 6128, 4, 38, 192, 24, 11, 5, 5963, 12, 196, 6052, 9, 5, 9911, 6, 8, 15, 5, 194, 10527, 11, 823, 457, 5, 247, 4, 38, 192, 24, 11, 3770, 54, 683, 6813, 7, 310, 24, 1522, 19, 42, 8560, 696, 53, 32, 122, 2882, 7, 1968, 49, 559, 5060, 15, 24, 4, 38, 192, 5, 7977, 11, 5, 2473, 9, 17179, 4211, 6, 1433, 11923, 7, 10645, 10, 14934, 88, 42, 4008, 36314, 1538, 232, 6, 54, 32, 12909, 11, 471, 78, 4, 38, 192, 24, 11, 5, 92, 16308, 937, 54, 20177, 414, 2018, 95, 141, 7163, 24, 64, 28, 4, 38, 192, 10, 7977, 11, 5, 15764, 9, 7476, 1791, 4, 286, 5, 78, 86, 10, 1647, 6, 4268, 4234, 4402, 63, 18426, 6, 19, 6791, 207, 3117, 24, 13, 1131, 6216, 4, 7737, 13, 18426, 34, 7408, 365, 332, 11, 5, 375, 367, 107, 1937, 4, 96, 15077, 6, 5, 78, 86, 21452, 553, 5, 864, 59, 18426, 6, 129, 316, 207, 9, 5, 1226, 21, 11, 4402, 4, 38, 192, 10, 7977, 14, 16, 6574, 1104, 2131, 566, 664, 82, 6, 53, 67, 924, 62, 566, 5, 1041, 8, 15198, 11, 127, 1159, 108, 334, 4, 83, 249, 1036, 38, 1145, 11, 2293, 16, 233, 9, 5, 7977, 6, 25, 32, 5, 15136, 9, 5, 1131, 8812, 6, 28132, 12557, 21712, 4, 38, 192, 24, 11, 5, 2419, 9, 205, 1041, 6, 32402, 12653, 49, 1074, 7, 120, 6150, 13, 49, 408, 480, 8, 11, 5, 408, 1235, 6, 215, 25, 5420, 6, 54, 439, 31, 519, 2993, 22107, 10, 186, 7, 95, 65, 50, 80, 10, 353, 4, 166, 216, 24, 351, 75, 6566, 33, 215, 5386, 775, 36, 368, 143, 913, 23, 70, 43, 11, 643, 6, 53, 99, 6150, 473, 116, 38, 192, 42, 1131, 3140, 7977, 11, 6167, 2127, 4, 9103, 18, 22107, 15220, 1131, 3140, 2309, 11, 3090, 479, 3687, 127, 4025, 6, 127, 1484, 8, 127, 964, 4, 38, 33, 190, 450, 5, 7977, 11, 127, 308, 284, 4, 83, 367, 107, 536, 6, 77, 38, 174, 127, 985, 38, 21, 3219, 5, 5674, 13, 10, 6717, 6, 38, 21, 1145, 19, 10, 251, 13787, 4, 22, 448, 41054, 734, 1917, 264, 39217, 11, 10, 457, 8026, 6, 457, 36631, 6645, 6328, 4, 264, 115, 6254, 190, 224, 5, 2136, 8, 69, 1263, 3820, 162, 19, 1403, 12, 417, 38766, 4, 1648, 25, 10, 3831, 313, 6, 3795, 64, 202, 146, 127, 32040, 1004, 1275, 8, 38637, 127, 2123, 19, 10, 881, 2136, 4, 125, 95, 94, 186, 79, 6017, 2294, 1084, 12, 3865, 3697, 1258, 8, 26, 6, 22, 100, 524, 2602, 9, 47, 15, 5, 1086, 3140, 631, 72, 38, 9010, 13, 5, 97, 12604, 7, 1874, 6, 53, 24, 399, 75, 4, 2978, 6, 79, 355, 6, 22, 1185, 1153, 1147, 10, 319, 9, 82, 54, 58, 3606, 72, 38, 218, 75, 206, 52, 56, 655, 56, 10, 1607, 101, 14, 65, 4, 497, 14, 1151, 6, 38, 794, 10, 7977, 14, 64, 836, 47, 7, 6941, 4, 20, 2136, 7977, 6, 606, 31, 5, 5862, 34633, 1182, 1020, 6, 7, 22, 15922, 198, 72, 38, 56, 127, 308, 1004, 198, 10, 891, 9, 107, 536, 6, 8, 23, 5, 86, 24, 21, 10, 20100, 317, 7, 946, 10, 8440, 737, 15, 1131, 3140, 4, 6206, 352, 143, 168, 503, 74, 2854, 7, 2662, 159, 8, 28, 7477, 15, 5, 5674, 4, 1648, 1484, 38, 1834, 7, 58, 11923, 7, 458, 49, 1652, 4, 85, 64, 28, 12792, 6, 38, 2435, 6, 7, 28, 15, 5, 235, 526, 9, 2866, 53, 15, 5, 1593, 526, 9, 14320, 4, 520, 52, 342, 5, 78, 22, 170, 196, 113, 6717, 15, 2384, 11, 830, 1014, 6, 38, 399, 75, 216, 114, 1268, 74, 1183, 84, 76, 3479, 803, 4, 1648, 3007, 6, 38, 399, 75, 190, 216, 114, 51, 74, 575, 4, 1534, 16062, 1030, 11, 110, 194, 116, 1801, 80, 107, 423, 6, 11, 22, 170, 196, 155, 60, 52, 32, 27541, 293, 7, 10, 7977, 11, 455, 7021, 4, 370, 40, 3068, 552, 19, 201, 13, 5, 14131, 9, 5, 78, 24793, 2033, 5154, 892, 15, 5, 304, 9, 3140, 13, 24679, 4, 370, 40, 972, 1484, 215, 25, 4640, 15932, 10197, 6, 41, 9370, 915, 15573, 6, 8, 26405, 2657, 6, 10, 1095, 12, 415, 12, 8361, 3795, 4, 252, 32, 5, 7063, 8, 6167, 2419, 9, 42, 7977, 480, 2793, 6, 1800, 8, 3606, 480, 20656, 7, 3264, 5, 754, 14, 10266, 14255, 12102, 747, 341, 7, 3951, 24679, 64, 28, 3007, 87, 5, 7482, 8364, 1495, 4, 4640, 15932, 10197, 823, 962, 6, 667, 7, 120, 357, 4, 370, 40, 192, 99, 16062, 269, 473, 7, 110, 2900, 6, 11, 16155, 699, 3156, 4, 152, 86, 198, 6, 47, 40, 1798, 31, 5, 3885, 9, 168, 2244, 22623, 352, 3565, 49, 477, 9, 1217, 6, 258, 1557, 8, 1172, 7028, 6, 8, 190, 5, 270, 9, 5, 315, 532, 4, 152, 16, 99, 10, 7977, 1326, 101, 4, 2486, 1131, 3140, 1142, 7173, 479, 520, 22, 170, 196, 132, 35, 20334, 30925, 113, 10843, 11, 494, 777, 6, 2278, 9338, 4434, 16149, 2614, 2047, 5, 235, 82, 58, 2494, 4, 1801, 237, 360, 423, 6, 16149, 2614, 829, 10, 1601, 11, 5, 7107, 37, 56, 57, 2445, 15, 13, 707, 107, 14, 1747, 1286, 752, 2846, 13, 39, 3140, 892, 4, 20, 752, 3380, 147, 16149, 2614, 74, 33, 7, 6925, 39, 3140, 16, 15, 5, 2894, 9, 13393, 4523, 11, 9238, 6, 5750, 4, 96, 14714, 9, 10, 6441, 7977, 6, 5, 931, 9, 557, 12, 8425, 3140, 89, 34, 1130, 389, 12, 12851, 11, 95, 5, 375, 76, 4, 5293, 117, 5021, 6, 52, 33, 2710, 9, 1283, 14, 5, 2846, 8, 323, 9, 5, 752, 168, 64, 1769, 1349, 10, 7977, 23, 10, 3845, 2877, 87, 52, 33, 648, 450, 4, 85, 21, 5, 496, 2534, 9, 404, 34043, 8, 32996, 6514, 34477, 14, 23498, 5, 557, 88, 10, 13306, 13, 17296, 6, 25, 157, 25, 8197, 5, 2504, 9, 580, 22568, 41199, 4, 252, 58, 67, 2149, 13, 5, 6344, 3685, 9, 25193, 13659, 30801, 8, 650, 40682, 4, 1944, 1800, 24793, 4094, 1767, 680, 5, 1050, 27392, 695, 6, 5, 34329, 2444, 3893, 8, 5, 29484, 8029, 11980, 4, 345, 32, 117, 8078, 9, 7721, 147, 5, 752, 168, 34, 57, 10, 24413, 9, 84, 285, 474, 782, 6, 8, 47, 115, 5848, 14, 1131, 3140, 74, 67, 6954, 25, 10, 22057, 915, 4, 158, 6357, 147, 1131, 3140, 115, 33, 913, 479, 345, 16, 122, 6177, 557, 88, 5, 304, 9, 3140, 14, 115, 913, 7281, 9, 1583, 9, 408, 8, 3362, 6, 217, 1416, 13, 1668, 6, 30239, 8, 11520, 18, 6, 7, 766, 10, 367, 4, 590, 6203, 7, 2400, 1937, 6, 3140, 115, 8908, 1888, 5, 1077, 13, 22274, 8, 11586, 7280, 5, 346, 9, 18305, 2400, 30563, 21532, 6, 61, 32, 5, 3968, 1303, 9, 2097, 868, 744, 11, 42, 247, 4, 287, 38, 4005, 420, 31, 12274, 4, 16071, 225, 6452, 11804, 463, 36, 495, 12, 4030, 469, 43, 8, 15405, 14725, 36, 495, 12, 4030, 3123, 238, 38, 1467, 402, 7116, 21, 2909, 4, 252, 58, 3872, 2838, 5, 527, 9, 5420, 20001, 118, 8, 10807, 97, 408, 4, 252, 58, 17977, 124, 5, 414, 52, 56, 1373, 31, 84, 656, 4941, 4, 252, 58, 8935, 3937, 154, 5, 801, 33975, 9, 5, 2195, 6, 8, 70, 9, 14, 21, 137, 5, 1194, 190, 554, 4, 345, 21, 41, 43635, 11465, 59, 106, 6, 8, 51, 2551, 11, 10, 20607, 7, 146, 10, 739, 14368, 11, 3140, 3114, 4, 252, 236, 3140, 7, 28, 5032, 3804, 12841, 4, 252, 236, 24, 122, 4, 252, 236, 3333, 7, 28, 441, 7, 30871, 24, 23, 11790, 4815, 70, 81, 5, 247, 4, 252, 236, 24, 122, 4, 252, 236, 557, 1932, 14313, 62, 7, 892, 5, 2195, 4, 252, 236, 24, 122, 4, 252, 236, 49, 2598, 2648, 23, 5, 194, 8, 632, 672, 7, 9630, 99, 144, 9, 5, 232, 6, 217, 5, 2286, 9, 5, 315, 532, 6, 33, 684, 13, 10, 251, 86, 35, 25249, 16, 10, 6150, 6, 14, 197, 28, 8069, 8, 3032, 101, 143, 97, 6150, 4, 178, 51, 236, 70, 9, 24, 122, 4, 38, 1240, 203, 9, 84, 1194, 4087, 106, 4, 38, 956, 7, 8736, 106, 14, 82, 6, 251, 137, 162, 50, 106, 6, 33, 57, 667, 7, 109, 171, 9, 209, 276, 383, 13, 843, 107, 6, 8, 56, 57, 3946, 358, 86, 4, 38, 9180, 106, 14, 3770, 33, 10, 543, 86, 1298, 1727, 15, 5, 696, 9, 3140, 53, 540, 9600, 2086, 106, 4, 38, 6835, 106, 358, 1149, 9, 5, 169, 4, 22, 713, 86, 40, 28, 430, 60, 14725, 27447, 174, 162, 25, 37, 3203, 66, 9, 5, 929, 4, 1534, 3140, 25, 1522, 25, 480, 50, 8788, 87, 480, 3766, 116, 38, 216, 141, 1365, 24, 16, 109, 1085, 142, 38, 222, 1085, 13, 350, 251, 4, 4624, 10, 205, 356, 23, 5, 414, 6, 11427, 2512, 8, 1067, 7, 5, 1484, 6, 54, 32, 747, 66, 9, 1735, 8, 465, 49, 1034, 11, 5, 1026, 9, 10, 2007, 2195, 4, 24446, 4395, 75, 185, 10, 737, 4, 85, 817, 1472, 4, 35671, 9866, 16, 8453, 4, 125, 6, 23, 103, 477, 6, 490, 1142, 109, 120, 7173, 4, 497, 103, 477, 6, 14883, 743, 109, 120, 8179, 4, 497, 103, 477, 6, 1537, 1472, 21720, 11791, 4, 407, 6, 259, 24, 16, 35, 166, 197, 25150, 1131, 3140, 4, 166, 197, 109, 24, 9852, 4, 178, 6, 52, 197, 109, 24, 122, 4, 361, 383, 7, 216, 59, 1030, 4728, 479, 50118, 47977, 35, 50118, 1640, 16256, 43, 20, 18426, 9, 1131, 3140, 11, 5, 315, 532, 34, 57, 10, 251, 12, 18536, 8, 4456, 696, 13, 1724, 4, 125, 5, 2625, 34, 57, 59, 549, 24, 197, 28, 1030, 50, 2439, 4, 20, 2625, 34, 57, 59, 549, 24, 197, 28, 22363, 50, 2439, 4, 20, 2625, 34, 57, 59, 549, 24, 197, 28, 22363, 50, 2439, 4, 20, 2625, 34, 57, 59, 549, 24, 197, 28, 22363, 50, 2439, 4, 50118, 133, 2625, 34, 57, 59, 549, 24, 197, 28, 22363, 50, 2439, 4, 50118, 133, 2625, 34, 57, 59, 549, 24, 197]])
        # if not (some[:,:input_ids.shape[1]].cpu() == input_ids.cpu()).all():
        #     pdb.set_trace()
        optimized = self.assistant_model.config.optimized
        # logging.error(f"{optimized=}")
        input_ids = input_ids.to(self.assistant_model.device)
        # logging.error(f"{input_ids.shape[1]=}")
        
        if self.different_tokenizers:
            convert_kwargs = {"src": self.target_tokenizer, "dest": self.assistant_tokenizer}

            if self.prev_tokens is None:
                draft_input_ids = convert_token_ids(input_ids, **convert_kwargs)
                self.prev_target_ids = input_ids
                self.prev_draft_ids = draft_input_ids
                new_cur_len = draft_input_ids.shape[-1]
            else:                
                # input_ids contains all target prompt input ids and some new target input ids
                if optimized and self.prev_target_ids.shape[1] > 20:
                    num_prev_target = self.prev_target_ids.shape[1]
                    target_lookbehind = min(self.target_lookbehind, input_ids.shape[1])
                    new_draft_ids = convert_token_ids(
                        input_ids[:, num_prev_target - target_lookbehind:], **convert_kwargs
                    )
                    draft_lookbehind_actual = min(new_draft_ids.shape[1] - 1, self.draft_lookbehind)
                    prev_draft_ids_tail = self.prev_draft_ids[:, -draft_lookbehind_actual:]
                    new_draft_input_ids = get_only_new_tokens(new_draft_ids, prev_draft_ids_tail)
                    draft_input_ids = torch.cat([self.prev_draft_ids, new_draft_input_ids], dim=-1)
                else:
                    draft_input_ids = convert_token_ids(input_ids, **convert_kwargs)

                min_draft_length = min(draft_input_ids.shape[1], self.prev_tokens.shape[1])
                draft_id_agreement = draft_input_ids[:, :min_draft_length] != self.prev_tokens[:, :min_draft_length]
                draft_id_agreement_nonzero = draft_id_agreement.nonzero()

                if draft_id_agreement_nonzero.shape[0] > 0:
                    mistake_index = draft_id_agreement_nonzero[0][1]
                    draft_input_ids = draft_input_ids[:, : mistake_index + 1]

                new_cur_len = draft_input_ids.shape[-1]
            # logging.error(f"{self.prev_target_ids.shape[1]=}")
        else:
            # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
            new_cur_len = input_ids.shape[-1]

        # max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        max_new_tokens = int(self.num_assistant_tokens)

        min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - new_cur_len), 0)
        if max_new_tokens == 0:
            return input_ids, None

        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            new_cache_size = new_cur_len - 1
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
            )  # the assistant does not have the token after the last match, hence the -1

            self.assistant_kwargs = _prepare_attention_mask(
                self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
            )
            self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: draft_input_ids,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
        }

        self.assistant_kwargs.pop("attention_mask", None)
        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

        if self.different_tokenizers:
            run_optimized = optimized and self.prev_draft_ids.shape[1] > 20
            if run_optimized:
                num_prev_draft = self.prev_draft_ids.shape[1]
                draft_lookbehind = min(self.draft_lookbehind, input_ids.shape[1])
                new_target_ids_from_window = convert_token_ids(
                    assistant_output.sequences[:, num_prev_draft - draft_lookbehind:],
                    src=self.assistant_tokenizer,
                    dest=self.target_tokenizer,
                )
                target_lookbehind_actual = min(new_target_ids_from_window.shape[1] - 1, self.target_lookbehind)
                # prev_target_ids_tail = self.prev_target_ids[:, -target_lookbehind_actual:]
                prev_target_ids_tail = input_ids[:, -target_lookbehind_actual:]
                new_target_input_ids = get_only_new_tokens(new_target_ids_from_window, prev_target_ids_tail)
                # new_target_ids = torch.cat([self.prev_target_ids, new_target_input_ids], dim=-1)
                new_target_ids = torch.cat([input_ids, new_target_input_ids], dim=-1)

            else:
                new_target_ids = convert_token_ids(
                    assistant_output.sequences,
                    src=self.assistant_tokenizer,
                    dest=self.target_tokenizer,
                )
                # new_target_ids = torch.cat([self.prev_target_ids, 
                #                             new_target_ids[:, self.prev_target_ids.shape[1]:]], dim=1)
                
                new_target_ids = torch.cat([input_ids, new_target_ids[:, input_ids.shape[1]:]], dim=1)
        else:
            new_target_ids = assistant_output.sequences
        # logging.error(f"{new_target_ids=}")
        # logging.error("===============\n")

        # 3. Update variables for the next round of candidate generation
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values
        self.prev_tokens = assistant_output.sequences

        # 4. Prepare variables for output
        candidate_logits = torch.stack(assistant_output.scores, dim=1)

        # if not (some[:,:1655].cpu() == new_target_ids[:,:1655].cpu()).all():
        #     pdb.set_trace()
        return new_target_ids, candidate_logits

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Adjust the max number of assistant tokens to use in the next iteration. This is a simple heuristic,
        # probably can be improved -- we want to balance the benefits of getting assistant tokens correct with the
        # cost of forecasting incorrect assistant tokens.
        if self.assistant_model.generation_config.num_assistant_tokens_schedule in {
            "heuristic",
            "heuristic_transient",
        }:
            if num_matches == int(self.num_assistant_tokens):
                self.num_assistant_tokens += 2.0
            else:
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)


class PromptLookupCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for prompt lookup generation. This class generates candidates by looking up
    likely continuations in the provided prompt (input_ids) itself.
    Read the following blog post for more information: https://github.com/apoorvumang/prompt-lookup-decoding

    Args:
        max_matching_ngram_size (`int`):
            The maximum ngram size to be considered for matching in the prompt
        num_output_tokens (`int`):
            The number of tokens to be output as candidate tokens.
        max_length (`int`):
            The number of total maximum tokens that can be generated. For decoder-only models that includes the prompt length.
            Defaults to 20, which is the max length used as default in generation config.
    """

    def __init__(
        self,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = None,
        max_length: int = 20,
    ):
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size if max_matching_ngram_size else 2
        self.max_length = max_length

        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(num_candidates, candidate_length)`: The candidate sequences to be tried.
        """
        input_length = input_ids.size(1)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        if self.max_length == input_length + 1:
            return input_ids, None

        chosen_ids = None
        match_found = False
        for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 0, -1):
            # Create sliding windows of size ngram_size
            windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

            # Convert ngram to a tensor for comparison
            ngram_tensor = input_ids[0, -ngram_size:]

            # Find where the windows match the ngram
            matches = (windows == ngram_tensor).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1]

            # Iterate through match indices to find a valid continuation
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.num_output_tokens
                end_idx = min(end_idx, input_length, self.max_length)

                if start_idx < end_idx:
                    chosen_ids = input_ids[0, start_idx:end_idx]
                    match_found = True
                    break
            if match_found:
                break

        if chosen_ids is None or len(chosen_ids) == 0:
            # In case we didn't find a match return the input sequence unchanged, reverts back to autoregressive decoding
            return input_ids, None

        # Now need extend input_ids with chosen_ids
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation expects logits as well, but we don't have those here, so returning None
        return candidate_input_ids, None

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Currently does nothing
        return


def _crop_past_key_values(model, past_key_values, max_length):
    """Crops the past key values up to a certain maximum length."""
    new_past = []
    if model.config.is_encoder_decoder:
        for idx in range(len(past_key_values)):
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :max_length, :],
                    past_key_values[idx][1][:, :, :max_length, :],
                    past_key_values[idx][2],
                    past_key_values[idx][3],
                )
            )
        past_key_values = tuple(new_past)
    # bloom is special
    elif "bloom" in model.__class__.__name__.lower() or (
        model.config.architectures is not None and "bloom" in model.config.architectures[0].lower()
    ):
        for idx in range(len(past_key_values)):
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :max_length],
                    past_key_values[idx][1][:, :max_length, :],
                )
            )
        past_key_values = tuple(new_past)
    # gptbigcode is too
    elif "gptbigcode" in model.__class__.__name__.lower() or (
        model.config.architectures is not None and "gptbigcode" in model.config.architectures[0].lower()
    ):
        if model.config.multi_query:
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :max_length, :]
        else:
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :, :max_length, :]
    elif isinstance(past_key_values, DynamicCache):
        past_key_values.crop(max_length)

    elif past_key_values is not None:
        for idx in range(len(past_key_values)):
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :max_length, :],
                    past_key_values[idx][1][:, :, :max_length, :],
                )
            )
        past_key_values = tuple(new_past)
    return past_key_values


def _prepare_attention_mask(model_kwargs: Dict[str, Any], new_length: int, is_encoder_decoder: bool) -> Dict[str, Any]:
    """Expands or crops the model's mask for decoding purposes, to the defined length"""

    mask_key = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
    if mask_key not in model_kwargs:
        return model_kwargs

    mask = model_kwargs[mask_key]
    mask_length_diff = new_length - mask.shape[1]

    if mask_length_diff < 0:
        model_kwargs[mask_key] = mask[:, :mask_length_diff]
    elif mask_length_diff > 0:
        model_kwargs[mask_key] = torch.cat([mask, mask.new_ones((mask.shape[0], mask_length_diff))], dim=-1)
    return model_kwargs


def _prepare_token_type_ids(model_kwargs: Dict[str, Any], new_length: int) -> Dict[str, Any]:
    """Expands or crops the model's token_type_ids for decoding purposes, to the defined length"""
    if "token_type_ids" not in model_kwargs or model_kwargs["token_type_ids"] is None:
        return model_kwargs

    token_type_ids = model_kwargs["token_type_ids"]
    final_token_type = token_type_ids[:, -1].unsqueeze(-1)
    type_length_diff = new_length - token_type_ids.shape[1]

    if type_length_diff < 0:
        token_type_ids = token_type_ids[:, :type_length_diff]
    elif type_length_diff > 0:
        token_type_copies = final_token_type.repeat(1, type_length_diff)
        model_kwargs["token_type_ids"] = torch.cat([model_kwargs["token_type_ids"], token_type_copies], dim=-1)
    return model_kwargs
