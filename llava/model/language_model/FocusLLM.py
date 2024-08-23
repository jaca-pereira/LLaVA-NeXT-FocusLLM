import numpy as np
import torch

from einops import einops
from torch.onnx.symbolic_opset11 import chunk
from transformers import DynamicCache, Cache, Qwen2Model, Qwen2Config
from typing import Optional, Union, Tuple, List

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast


logger = logging.get_logger(__name__)

class FocusLLMModel(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.last_attention = None

    def process_attention_scores(self, attention_scores, topk_ratio=None, smooth_forward_segments=0):
        """Process attention scores to obtain top-k or bottom-k indices."""
        if topk_ratio is not None:
            # Get bottom-k indices for segment pruning
            ratio = int(self.image_video_tokens * topk_ratio)
            indices = attention_scores.topk(ratio, largest=False, dim=-1).indices.sort(dim=-1).values
        else:
            if self.config.use_cpu:
                attention_scores = attention_scores.to(self.device)
            indices = [chunk.flatten().topk(self.image_video_tokens, largest=True, dim=-1).indices.sort(dim=-1).values
                       for chunk in attention_scores.chunk(smooth_forward_segments)]
            if self.config.use_cpu:
                attention_scores = attention_scores.to('cpu')
                indices = [chunk.to('cpu') for chunk in indices]
        return indices

    def update_hidden_states(self, hidden_states, topk_idx):
        """Update hidden states with selected image/video tokens."""
        hidden_states = [chunk for chunk in hidden_states.chunk(len(topk_idx))]
        for i, idx in enumerate(topk_idx):
            hidden_states_image_or_video = hidden_states[i][:, self.modal_token_index:self.modal_token_index + self.image_video_tokens, :]
            hidden_states_image_or_video = einops.rearrange(hidden_states_image_or_video, 'b n d -> 1 (b n) d')[:, idx]
            hidden_states[i] = torch.cat(
                (
                    hidden_states[i][0:1, :self.modal_token_index, :],
                    hidden_states_image_or_video,
                    hidden_states[i][0:1, self.modal_token_index + self.image_video_tokens:, :]
                ), dim=-2)
        return torch.cat(hidden_states) if len(hidden_states) > 1 else hidden_states[0]

    def update_past_key_values(self, past_key_values, topk_idx):
        """Update past key and value caches for the selected image/video tokens."""
        if isinstance(past_key_values, DynamicCache):
            is_cache = True
            new_past_key_values = []
            for i in range(len(past_key_values.key_cache[0])):
                new_past_key_values.append(DynamicCache())
                for layer_idx in range(len(past_key_values.key_cache)):
                    new_past_key_values[i].key_cache.append(
                        past_key_values.key_cache[layer_idx][i:i+1])
                    new_past_key_values[i].value_cache.append(
                        past_key_values.value_cache[layer_idx][i:i+1])
            past_key_values = new_past_key_values

        else:
            is_cache = False
        chunks = [len(chunk) for chunk in torch.ones((len(past_key_values))).chunk(len(topk_idx))]
        chunks = [0] + torch.cumsum(torch.tensor(chunks), dim=0).tolist()
        past_key_values = [past_key_values[chunks[i]:chunks[i + 1]] for i in range(len(chunks) - 1)]

        for i, past_key_values_i in enumerate(past_key_values):
            key_cache_img_vid = [None] * len(past_key_values_i[0].key_cache)
            value_cache_img_vid = [None] * len(past_key_values_i[0].value_cache)
            for layer_idx in range(len(past_key_values_i[0].key_cache)):
                key_cache_img_vid[layer_idx] = past_key_values_i[0].key_cache[layer_idx][..., self.modal_token_index:self.modal_token_index + self.image_video_tokens, :]
                value_cache_img_vid[layer_idx] = past_key_values_i[0].value_cache[layer_idx][..., self.modal_token_index:self.modal_token_index + self.image_video_tokens, :]
                for past_key_values_j in past_key_values_i[1:]:
                    key_cache_img_vid[layer_idx] = torch.cat(
                        (
                            key_cache_img_vid[layer_idx],
                            past_key_values_j.key_cache[layer_idx][..., self.modal_token_index:self.modal_token_index + self.image_video_tokens, :]
                        ),
                        dim=-2
                    )
                    value_cache_img_vid[layer_idx] = torch.cat(
                        (
                            value_cache_img_vid[layer_idx],
                            past_key_values_j.value_cache[layer_idx][..., self.modal_token_index:self.modal_token_index + self.image_video_tokens, :]
                        ),
                        dim=-2
                    )

                key_cache_img_vid[layer_idx] = key_cache_img_vid[layer_idx][..., topk_idx[i], :]
                value_cache_img_vid[layer_idx] = value_cache_img_vid[layer_idx][..., topk_idx[i], :]
                key_cache_img_vid[layer_idx] = torch.cat(
                    (
                        past_key_values_i[0].key_cache[layer_idx][..., :self.modal_token_index, :],
                        key_cache_img_vid[layer_idx],
                        past_key_values_i[0].key_cache[layer_idx][..., self.modal_token_index + self.image_video_tokens:, :]
                    ),
                    dim=-2
                )
                value_cache_img_vid[layer_idx] = torch.cat(
                    (
                        past_key_values_i[0].value_cache[layer_idx][..., :self.modal_token_index, :],
                        value_cache_img_vid[layer_idx],
                        past_key_values_i[0].value_cache[layer_idx][..., self.modal_token_index + self.image_video_tokens:, :]
                    ),
                    dim=-2
                )
            past_key_values[i] = DynamicCache()
            past_key_values[i].key_cache = key_cache_img_vid
            past_key_values[i].value_cache = value_cache_img_vid

        if is_cache:
            new_past_key_values = DynamicCache()
            new_past_key_values.key_cache = [torch.cat([past_key_values[j].key_cache[i] for j in range(len(past_key_values))]) for i in range(len(past_key_values[0].key_cache))]
            new_past_key_values.value_cache = [torch.cat([past_key_values[j].value_cache[i] for j in range(len(past_key_values))]) for i in range(len(past_key_values[0].value_cache))]
            past_key_values = new_past_key_values
            del new_past_key_values
        del key_cache_img_vid, value_cache_img_vid
        return past_key_values

    def process_layer(self, decoder_layer, hidden_states, attention_mask, position_ids, past_key_values, layer_idx,
                      device, output_attentions, output_hidden_states, use_cache):
        """Process a single decoder layer, updating hidden states and caching."""
        if self.config.use_sequential:
            # Sequential processing for each hidden state slice
            past_key_values = [DynamicCache() for _ in range(len(hidden_states))] if isinstance(past_key_values, DynamicCache) else past_key_values
            layer_outputs = [decoder_layer(hidden_states[i].unsqueeze(0).to(device),
                                           attention_mask[i].unsqueeze(0).to(device),
                                           position_ids[i].unsqueeze(0).to(device), past_key_values[i],
                                           output_attentions=output_attentions,
                                           use_cache=use_cache) for i in range(len(hidden_states))]
            hidden_states = torch.cat([lo[0].to('cpu' if self.config.use_cpu else None) for lo in layer_outputs], dim=0)
            last_attention = torch.cat([lo[1].to('cpu' if self.config.use_cpu else None) for lo in layer_outputs],
                                                dim=0) if layer_idx in self.config.focus_layers - 1 else None
            return hidden_states, last_attention, past_key_values
        else:
            # Parallel processing
            layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask,
                                          position_ids=position_ids, past_key_value=past_key_values,
                                          output_attentions=output_attentions, use_cache=use_cache)
            last_attention = layer_outputs[1] if layer_idx in self.config.focus_layers - 1 else None
            return layer_outputs[0], last_attention, past_key_values

    def focus_llm_forward(self, hidden_states, attention_mask, position_ids, inputs_embeds, past_key_values, device, output_attentions, output_hidden_states, use_cache):
        """Reimplementation of the forward pass for Mistral LLM with focus on efficiency."""
        seq_length = hidden_states.shape[1]
        if self.config.focus_llm and seq_length > 1:
            for decoder_layer in self.layers:
                layer_idx = decoder_layer.self_attn.layer_idx
                if layer_idx in self.config.focus_layers:
                    if self.config.plot_sys_user_prompt_sim:
                        hidden_states_system = hidden_states[..., :self.modal_token_index, :]
                        hidden_states_user = hidden_states[..., self.modal_token_index + self.image_video_tokens:, :]
                        hidden_states_image_video = hidden_states[...,
                                                    self.modal_token_index:self.modal_token_index + self.image_video_tokens,
                                                    :]
                        plot_sim_and_tsne(hidden_states_system, video_name=self.config.video_name, name="system",
                                          layer_num=layer_idx, frame_num=seq_length * NUM_FRAMES)
                        plot_sim_and_tsne(hidden_states_user, video_name=self.config.video_name, name="user",
                                          layer_num=layer_idx, frame_num=seq_length * NUM_FRAMES)
                        plot_sim_and_tsne(hidden_states_image_video, video_name=self.config.video_name, name="video",
                                          layer_num=layer_idx, frame_num=seq_length * NUM_FRAMES)

                    image_attention_score = last_attention.mean(dim=1)[..., -1,
                                            self.modal_token_index:self.modal_token_index + self.image_video_tokens]

                    if not self.config.segment_pruning and layer_idx == self.config.focus_layers[-1]:
                        bottom_attention_rank_index = self.process_attention_scores(image_attention_score,
                                                                               topk_ratio=self.config.ratio)
                        attention_mask[..., bottom_attention_rank_index] = attention_mask[0, 0, 0, -1].item()
                        attention_mask[..., bottom_attention_rank_index, :] = attention_mask[0, 0, 0, -1].item()
                        if self.config.reforward:
                            hidden_states = inputs_embeds
                            past_key_values = DynamicCache()
                        position_ids = position_ids[0:1]
                        break
                    else:
                        smooth_forward_segments = \
                        self.config.smooth_forward_segments[np.where(self.config.focus_layers == layer_idx)][
                            0]
                        topk_idx = self.process_attention_scores(image_attention_score,
                                                                 smooth_forward_segments=smooth_forward_segments)
                        if layer_idx == self.config.focus_layers[-1] and self.config.reforward:
                            hidden_states = self.update_hidden_states(inputs_embeds, topk_idx)
                            past_key_values = DynamicCache()
                            attention_mask = attention_mask[:1]
                            position_ids = position_ids[:1]
                            break
                        else:
                            hidden_states = self.update_hidden_states(hidden_states, topk_idx)
                            past_key_values = self.update_past_key_values(past_key_values, topk_idx)
                            attention_mask = attention_mask[:len(hidden_states)]
                            position_ids = position_ids[:len(hidden_states)]
                            if layer_idx == self.config.focus_layers[-1]:
                                if not isinstance(past_key_values, DynamicCache):
                                    past_key_values = past_key_values[0]
                                break

                # Process each layer and update hidden states and attention if required
                hidden_states, last_attention, past_key_values = self.process_layer(decoder_layer, hidden_states, attention_mask, position_ids, past_key_values, layer_idx, device, output_attentions, output_hidden_states, use_cache)

        return hidden_states.to(device), attention_mask.to(device), position_ids.to(device), past_key_values

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        if self.config.original_seq_length == -1:
            self.config.original_seq_length = seq_length

        if seq_length > 0 and seq_length < self.config.original_seq_length and self.config.segment_pruning:
            position_ids = position_ids[:, -1:]
            input_ids = input_ids[:, -1:]
            attention_mask = attention_mask[:, -1:]
            batch_size, seq_length = input_ids.shape
        if seq_length > 1 and self.config.segment_pruning:
            if self.config.original_seq_length is None:
                self.original_seq_length = seq_length
            segment_batch_size = self.total_image_video_tokens//self.image_video_tokens
            user_seq_length = seq_length - self.modal_token_index - self.total_image_video_tokens
            attention_mask = attention_mask[:, :self.modal_token_index + self.image_video_tokens + user_seq_length].repeat(segment_batch_size, 1)
            position_ids = position_ids[:, :self.modal_token_index + self.image_video_tokens + user_seq_length].repeat(segment_batch_size, 1)
            inputs_embeds_image_video = inputs_embeds[:, self.modal_token_index:self.modal_token_index+self.total_image_video_tokens]
            inputs_embeds_image_video = einops.rearrange(inputs_embeds_image_video, 'b (s n) d -> (b s) n d', s=segment_batch_size)
            inputs_embeds = torch.cat(
                (
                    inputs_embeds[:, :self.modal_token_index, :].repeat(segment_batch_size, 1, 1),
                    inputs_embeds_image_video,
                    inputs_embeds[:, self.modal_token_index+self.total_image_video_tokens:, :].repeat(segment_batch_size, 1, 1)
                ), dim=1
            )
            batch_size, seq_length, _ =  inputs_embeds.shape

        if self.config.segment_pruning and seq_length == 1:
            position_ids = position_ids - (self.total_image_video_tokens - self.image_video_tokens)
            attention_mask = attention_mask[:, :position_ids[0]+1]

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        """if seq_length > 1 and self.config.segment_pruning and position_ids.shape[0] != inputs_embeds.shape[0]:
            position_ids = position_ids.repeat(inputs_embeds.shape[0], 1)"""

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states and not self.config.segment_pruning else None
        all_self_attns = () if output_attentions and not self.config.segment_pruning else None
        next_decoder_cache = None

        ################ new forward pass loop for getting best idx ###########################
        hidden_states, attention_mask, position_ids, past_key_values = self.focus_llm_forward(hidden_states,
                                                                                              attention_mask,
                                                                                              position_ids,
                                                                                              inputs_embeds,
                                                                                              past_key_values,
                                                                                              hidden_states.device if self.config.use_cpu else None,
                                                                                              output_attentions,
                                                                                              output_hidden_states,
                                                                                              use_cache)

        ##########################################################################################
        iterated_layers = self.layers[self.config.focus_layers[
                                          -1]:] if not self.config.reforward and self.config.focus_llm and seq_length > 1 else self.layers
        for decoder_layer in iterated_layers:
            if output_hidden_states and not self.config.segment_pruning:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions and not self.config.segment_pruning:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states and not self.config.segment_pruning:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

