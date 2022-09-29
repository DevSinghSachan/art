# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron global variables."""

import os
import sys
import time
import csv

import torch

from megatron.tokenizer import build_tokenizer
from .arguments import parse_args
from transformers import T5Tokenizer, T5ForConditionalGeneration


_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None
_GLOBAL_T0_TOKENIZER = None
_GLOBAL_T0_MODEL = None
_GLOBAL_WIKIPEDIA_EVIDENCE = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_TIMERS = None


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_t0_tokenizer():
    """Return T0 tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_T0_TOKENIZER, 'T0-tokenizer')
    return _GLOBAL_T0_TOKENIZER


def get_t0_model():
    """Return T0 model."""
    _ensure_var_is_initialized(_GLOBAL_T0_MODEL, 'T0-model')
    return _GLOBAL_T0_MODEL

def get_wikipedia_evidence():
    """Return T0 model."""
    _ensure_var_is_initialized(_GLOBAL_WIKIPEDIA_EVIDENCE, 'wikipedia-evidence-from-DPR-paper')
    return _GLOBAL_WIKIPEDIA_EVIDENCE


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS


def set_global_variables(extra_args_provider=None, args_defaults={},
                         ignore_unknown_args=False):
    """Set args, tokenizer, tensorboard-writer, and timers."""
    args = _parse_args(extra_args_provider=extra_args_provider,
                       defaults=args_defaults,
                       ignore_unknown_args=ignore_unknown_args)
    _ = _build_tokenizer(args)

    if args.initialize_t0_model_tokenizer_evidence:
        _ = _build_t0_tokenizer(args)
        _ = _build_t0_model(args)
        _ = _load_wikipedia_evidence(args)

    _set_tensorboard_writer(args)
    _set_timers()


def _parse_args(extra_args_provider=None, defaults={},
                ignore_unknown_args=False):
    """Parse entire arguments."""
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    _GLOBAL_ARGS = parse_args(extra_args_provider=extra_args_provider,
                              defaults=defaults,
                              ignore_unknown_args=ignore_unknown_args)
    return _GLOBAL_ARGS


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def _load_wikipedia_evidence(args):
    """Load the DPR wikipedia evidence file"""
    global _GLOBAL_WIKIPEDIA_EVIDENCE
    _ensure_var_is_not_initialized(_GLOBAL_WIKIPEDIA_EVIDENCE, 'wikipedia-evidence-from-DPR-paper')
    _GLOBAL_WIKIPEDIA_EVIDENCE = process_samples_from_single_path(args)
    return _GLOBAL_WIKIPEDIA_EVIDENCE


def _build_t0_tokenizer(args):
    """Initialize T0 tokenizer."""
    global _GLOBAL_T0_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_T0_TOKENIZER, 'T0-tokenizer')
    _GLOBAL_T0_TOKENIZER = T5Tokenizer.from_pretrained(args.hf_model_name)
    return _GLOBAL_T0_TOKENIZER


def _build_t0_model(args):
    """Initialize T0 model."""
    global _GLOBAL_T0_MODEL
    _ensure_var_is_not_initialized(_GLOBAL_T0_MODEL, 'T0-model')
    _GLOBAL_T0_MODEL = T5ForConditionalGeneration.from_pretrained(args.hf_model_name,
                                                                  torch_dtype=torch.bfloat16 if args.t0_model_in_bf16 else torch.float32)

    # if args.t0_model_in_bf16:
    #     _GLOBAL_T0_MODEL = _GLOBAL_T0_MODEL.bfloat16()

    for param in _GLOBAL_T0_MODEL.parameters():
        param.requires_grad = False
    return _GLOBAL_T0_MODEL


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and \
       args.tensorboard_dir and args.rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '_time', value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(string, flush=True)
        else:
            print(string, flush=True)


def process_samples_from_single_path(args):

    if args.local_rank == 0:
        print(' > Processing {} ...'.format(args.evidence_data_path))
    total = 0
    id2text = []

    with open(args.evidence_data_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # skip the headers

        # Fill some random text tuple in the first index
        id2text.append(("text", "title"))

        for row in reader:
            # file format: doc_id, doc_text, title
            doc_id = int(row[0])
            text = row[1]
            title = row[2]

            # doc_id is specified by the index of the list
            id2text.append((text, title))

            total += 1
            if total % 1000000 == 0:
                if args.local_rank == 0:
                    print('  > processed {} rows so far ...'.format(total))

    if args.local_rank == 0:
        print(' >> processed {} samples.'.format(total))

    return id2text
