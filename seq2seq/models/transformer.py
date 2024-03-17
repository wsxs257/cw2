#   Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017) <https://arxiv.org/abs/1706.03762>`.
#   This is a basic version intended for demonstration ONLY as part of INFR11157 course (NLU+).
#   Tom Sherborne, University of Edinburgh, January 2020
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture
from seq2seq.models.transformer_helper import TransformerEncoderLayer, TransformerDecoderLayer, PositionalEmbedding, generate_embedding, fill_with_neg_inf

DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512


@register_model('transformer')
class TransformerModel(Seq2SeqModel):
    """
    Transformer Model Class. Inherits from Seq2SeqModel and calls TransformerEncoder and TransformerDecoder submodels.
    """
    def __init__(self,
                 args,
                 encoder,
                 decoder):
        super().__init__(encoder, decoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D', help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N', help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', help='num decoder attention heads')
        parser.add_argument('--no-scale-embedding', action='store_true', help='if True, dont scale embeddings')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """Construct model. """
        base_architecture(args)

        if getattr(args, 'max_source_positions', None) is None:
            args.max_src_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_tgt_positions = DEFAULT_MAX_TARGET_POSITIONS

        # Transformer Encoder
        encoder = TransformerEncoder(args, src_dict)
        decoder = TransformerDecoder(args, tgt_dict)
        return cls(args, encoder, decoder)


class TransformerEncoder(Seq2SeqEncoder):
    """ Defines an encoder class. """

    def __init__(self,
                 args,
                 dictionary):

        super().__init__(dictionary)

        self.dropout = args.dropout
        self.embed_dim = args.encoder_embed_dim
        self.padding_idx = dictionary.pad_idx
        self.max_src_positions = args.max_src_positions
        self.embedding = generate_embedding(len(dictionary), self.embed_dim, dictionary.pad_idx)
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.embed_dim)

        self.embed_positions = PositionalEmbedding(
            self.embed_dim, padding_idx=self.padding_idx, init_size=self.max_src_positions + self.padding_idx + 1
        )

        self.layers = nn.ModuleList([])

        # Generate N identical Encoder Layers
        self.layers.extend([
            TransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])

    def forward(self, src_tokens, src_lengths):
        # Embed tokens indices
        embeddings = self.embed_scale * self.embedding(src_tokens)

        # Clone for output state
        src_embeddings = embeddings.clone()

        '''
        ___QUESTION-6-DESCRIBE-A-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  What is the purpose of the positional embeddings in the encoder and decoder? 
        
        The purpose of the positional embeddings in the encoder and decoder is to add information about the position 
        of each token to the input embeddings of the Transformer. Transformer operates on its inputs without taking into 
        account the sequence ordering (permutation equivariant).Positional embeddings solve this by providing additional 
        information about the position of each token in the sequence.Positional embeddings make model's predictions based 
        on the order of elements in the sequence.
        '''
        # embedding.size = [batch_size, src_time_steps, num_features]
        embeddings += self.embed_positions(src_tokens)
        '''
        ___QUESTION-6-DESCRIBE-A-END___
        '''
        forward_state = F.dropout(embeddings, p=self.dropout, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features]
        forward_state = forward_state.transpose(0, 1)

        # Compute padding mask for attention
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # Forward pass through each Transformer Encoder Layer
        for layer in self.layers:
            forward_state = layer(state=forward_state, encoder_padding_mask=encoder_padding_mask)

        return {
            "src_out": forward_state,   # [src_time_steps, batch_size, num_features]
            "src_embeddings": src_embeddings,   # [batch_size, src_time_steps, num_features]
            "src_padding_mask": encoder_padding_mask,   # [batch_size, src_time_steps]
            "src_states": []    # List[]
        }


class TransformerDecoder(Seq2SeqDecoder):
    """ Defines an decoder class. """
    def __init__(self,
                 args,
                 dictionary):

        super().__init__(dictionary)
        self.dropout = args.dropout
        self.embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_embed_dim
        self.padding_idx = dictionary.pad_idx
        self.max_tgt_positions = args.max_tgt_positions

        self.embedding = generate_embedding(len(dictionary), self.embed_dim, dictionary.pad_idx)
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.embed_dim)

        self.embed_positions = PositionalEmbedding(
            self.embed_dim, padding_idx=self.padding_idx, init_size=self.max_tgt_positions + self.padding_idx + 1
        )

        self.layers = nn.ModuleList([])

        # Generate N identical Decoder Layers
        self.layers.extend([
            TransformerDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])

        self.embed_out = nn.Linear(self.output_embed_dim, len(dictionary))

        nn.init.normal_(self.embed_out.weight, mean=0, std=self.output_embed_dim ** -0.5)

    def forward(self, tgt_inputs, encoder_out=None, incremental_state=None, features_only=False):
        # Embed positions
        positions = self.embed_positions(tgt_inputs, incremental_state=incremental_state)

        # Incremental decoding only needs the single previous token
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        forward_state = self.embed_scale * self.embedding(tgt_inputs)
        forward_state += positions
        forward_state = F.dropout(forward_state, p=self.dropout, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [tgt_time_steps, batch_size, num_features]
        forward_state = forward_state.transpose(0, 1)

        # Generate padding mask
        self_attn_padding_mask = tgt_inputs.eq(self.padding_idx) if tgt_inputs.eq(self.padding_idx).any() else None

        # Forward pass through each Transformer Decode Layer
        attn_state = None
        inner_states = [forward_state]

        for layer_idx, layer in enumerate(self.layers):
            is_attention_layer = layer_idx == len(self.layers) - 1
            encoder_state = encoder_out['src_out'] if encoder_out is not None else None
            '''
            ___QUESTION-6-DESCRIBE-B-START___
            1.  Add tensor shape annotation to each of the output tensor
            2.  What is the purpose of self_attn_mask? 
            The purpose of self_attn_mask is to prevent the network from “seeing” / computing attention scores for 
            future input words. Hence, the network can only attend to previous and current input words when making 
            the current prediction. 
            
            3.  Why do we need it in the decoder but not in the encoder?
            The encoder processes the entire input source sequence at once and allows each position to attend to all 
            positions within the same sequence, which aids in understanding the full source context. In the training 
            phase, the transformer uses the Teacher Forcing training strategy. The decoder uses the entire target 
            sentence as input, so the entire target sequence is known. However, the decoder should generate a sequence 
            word-by-word. To simulate sequence generation, the model should only rely on the elements that have already 
            been generated and the information from the input sequence. Therefore, through masking, we can prevent 
            positions from attending to subsequent positions. This masking prevents the model from accessing future 
            positions when calculating Self-Attention, ensuring that the output at the current position can only depend 
            on the previous outputs and the input sequence. This ensures that the model is auto-regressive, avoiding 
            "peeking" at future information.
            
             
            4.  Why do we not need a mask for incremental decoding?
            We do not need a mask for incremental decoding because during incremental decoding we decode the target 
            sequence token-by-token. We do not provide the inputs with future time steps to the decoder.
            Model can not attending to future tokens because those tokens have not been generated.
            Thus, we do not need to mask them out.
            '''

            # self_attn_mask.size = [tgt_time_steps, tgt_time_steps] if incremental_state is None, otherwise = None
            self_attn_mask = self.buffered_future_mask(forward_state) if incremental_state is None else None
            '''
            ___QUESTION-6-DESCRIBE-B-END___
            '''

            forward_state, layer_attn = layer(state=forward_state,
                                              encoder_out=encoder_state,
                                              self_attn_mask=self_attn_mask,
                                              self_attn_padding_mask=self_attn_padding_mask,
                                              need_attn=is_attention_layer,
                                              need_head_weights=is_attention_layer)
            inner_states.append(forward_state)

            if layer_attn is not None and is_attention_layer:
                attn_state = layer_attn.float()

        if attn_state is not None:
            attn_state = attn_state.mean(dim=0)

        forward_state = forward_state.transpose(0, 1)

        # Project into output layer
        if not features_only:
            '''
            ___QUESTION-6-DESCRIBE-C-START___
            1.  Why do we need a linear projection after the decoder layers? 
            The linear projection transforms the decoder's output dimensionality to match the vocabulary size. 
            It convert the feature representations into a space where each dimension corresponds to a vocabulary token.
            So model can make predictions of the next word from the vocabulary.
            
            2.  What would the output represent if features_only=True?
            If features_only is set to True, the decoder outputs the feature representations (hidden state) directly 
            after processing through the decoder layers but before the final linear projection. 
            These features can be used as inputs to new layers to fine-tune new tasks rather than being used directly for
            token prediction.
            '''
            forward_state = self.embed_out(forward_state)
            '''
            ___QUESTION-6-DESCRIBE-C-END___
            '''
        return forward_state, {
            "attn_state": attn_state,
            "inner_states": inner_states
        }

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (not hasattr(self, '_future_mask')) or self._future_mask is None or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    # Enc-Dec params [mostly tied for simplicity]
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', args.encoder_layers)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', args.encoder_attention_heads)

    # Dropout and activation
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.2)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
