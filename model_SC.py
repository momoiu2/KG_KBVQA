'''
Adapted from https://github.com/huggingface/transformers
'''

from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG
import copy
import os
import warnings
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
from torch_geometric.nn import GCNConv
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(output_dim)

        # self.g_out = nn.Linear(output_dim,768)

    def forward(self, x_list, edge_index_list):
        outputs = []
        for x, edge_index in zip(x_list, edge_index_list):
            # if x.size() == torch.Size([0]) or x.size() == torch.Size([1, 0]):
            #     x = torch.ones(1, 300)
            #     edge_index = torch.tensor([0, 1])

            out = self.conv1(x, edge_index).relu()
            out = self.norm1(out)
            out = self.conv2(out, edge_index)
            # out = self.norm2(out)
            # print(out)
            # out = self.g_out(out)
            outputs.append(out)
        return torch.stack(outputs)
    # def forward(self, graph, features):
    #     x = self.conv1(graph, features)
    #     x = F.relu(x)
    #     x = self.conv2(graph, x)
    #     return x.unsqueeze(0)
class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size, padding_idx, save_dir, vot_num, alpha):
        super().__init__(config)
        self.model_dim = config.d_model
        self.vot_num = vot_num
        self.alpha = alpha
        self.padding_idx = padding_idx

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_num, self.patch_dim = patch_size
        self.gcnm_model = GCNModel(input_dim=300, hidden_dim=256, output_dim=256)
        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size,
                                                     vdim=config.hidden_size, num_heads=1, batch_first=True)
        # self.gate_dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        self.kg_dense = nn.Linear(256, config.d_model)
        self.gate_dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.gate_dense1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.gate_dense2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.gha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size,
                                                     vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            image_ids=None,
            data=None,
            # graph=None,
            # features=None,
            edge_attr=None,
            edge_index=None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        decoder_input_ids_base = decoder_input_ids
        # decoder_attention_mask_base=decoder_attention_mask
        decoder_inputs_embeds_base = decoder_inputs_embeds
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=True,  # 设置这里 output_attentions
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        gcnm_out = self.gcnm_model(data, edge_index)
        hidden_states = encoder_outputs[0]  # (1,512,768) 这里存在问题和外部知识
        # print("hidden_states")
        # print(hidden_states)
        image_embedding = self.image_dense(image_ids)  # (1,49,768)这里是图片特征
        # print("image_embedding")
        # print(image_embedding)
        image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding)  # (1,512,768)
        merge = torch.cat([hidden_states, image_att], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        hidden_states1 = (1 - gate) * hidden_states + gate * image_att
        # print(hidden_states)
        # gcnm_out = self.kgnorm(gcnm_out)
        kg_embedding = self.kg_dense(gcnm_out)
        # print("kg_embedding")
        # kg_embedding = gcnm_out
        # print(kg_embedding)
        kg_att, _ = self.gha_layer(hidden_states, kg_embedding, kg_embedding)
        merge1 = torch.cat([hidden_states, kg_att], dim=-1)
        gate1 = self.sigmoid(self.gate_dense1(merge1))
        hidden_states2 = (1 - gate1) * hidden_states + gate1 * kg_att

        merge2 = torch.cat([hidden_states1, hidden_states2], dim=-1)
        gate2 = self.sigmoid(self.gate_dense2(merge2))
        hidden_states = (1 - gate2) * hidden_states1 + gate2 * hidden_states2

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        all_logits = []

        for _ in range(self.vot_num):
            # Decode
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = decoder_outputs[0]
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim ** -0.5)

            lm_logits = self.lm_head(sequence_output)
            all_logits.append(lm_logits)

        # voting
        stacked_logits = torch.stack(all_logits, dim=0)
        mean_logits = torch.mean(stacked_logits, dim=0)
        stddev_logits = torch.std(stacked_logits, dim=0)
        weights = 1 / (1 + stddev_logits)
        weighted_mean_logits = torch.sum(weights * stacked_logits, dim=0) / torch.sum(weights, dim=0)
        alpha = self.alpha
        lm_logits = alpha * mean_logits + (1 - alpha) * weighted_mean_logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )