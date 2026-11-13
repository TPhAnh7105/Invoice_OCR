"""Model definitions extracted from invoice-ocr-with-metrics.ipynb.
Regenerate with: python tools/export_demo_models.py
PAN and OCR live in separate modules to avoid notebook name collisions.
"""
import math
import numpy as np
import torch
from torch import nn
from torchvision import models
from torch.nn.functional import softmax, log_softmax
from PIL import Image
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()
        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]
        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])
        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        conv = self.relu(x)
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv

def Resnet50(ss, hidden):
    return ResNet(3, hidden, BasicBlock, [1, 2, 5, 3])

class Vgg(nn.Module):

    def __init__(self, name, ss, ks, hidden, pretrained=True, dropout=0.5):
        super(Vgg, self).__init__()
        if name == 'vgg11_bn':
            cnn = models.vgg11_bn(pretrained=pretrained)
        elif name == 'vgg19_bn':
            cnn = models.vgg19_bn(pretrained=pretrained)
        pool_idx = 0
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1
        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        """
        Shape: 
            - x: (N, C, H, W)
            - output: (W, N, C)
        """
        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv

def vgg11_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg11_bn', ss, ks, hidden, pretrained, dropout)

def vgg19_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg19_bn', ss, ks, hidden, pretrained, dropout)

class CNN(nn.Module):

    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()
        if backbone == 'vgg11_bn':
            self.model = vgg11_bn(**kwargs)
        elif backbone == 'vgg19_bn':
            self.model = vgg19_bn(**kwargs)
        elif backbone == 'resnet50':
            self.model = Resnet50(**kwargs)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True

class LanguageTransformer(nn.Module):

    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Shape:
            - src: (W, N, C)
            - tgt: (T, N) 
            - src_key_padding_mask: (N, S)
            - tgt_key_padding_mask: (N, T)
            - memory_key_padding_mask: (N, S)
            - output: (N, T, E)
            
        """
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(src.device)
        src = self.pos_enc(src * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        return self.fc(output)

    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_encoder(self, src):
        src = self.pos_enc(src * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src)
        return memory

    def forward_decoder(self, tgt, memory):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        return (self.fc(output), memory)

    def expand_memory(self, memory, beam_size):
        memory = memory.repeat(1, beam_size, 1)
        return memory

    def get_memory(self, memory, i):
        memory = memory[:, [i], :]
        return memory

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(0)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(-1).expand(x.size()[:2])
        x = x + self.pos_embed(pos)
        return self.dropout(self.layernorm(x))

class LayerNorm(nn.Module):
    """A layernorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, d_model, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class VietOCR(nn.Module):

    def __init__(self, vocab_size, backbone, cnn_args, transformer_args, seq_modeling='transformer'):
        super(VietOCR, self).__init__()
        self.cnn = CNN(backbone, **cnn_args)
        self.seq_modeling = seq_modeling
        if seq_modeling == 'transformer':
            self.transformer = LanguageTransformer(vocab_size, **transformer_args)
        elif seq_modeling == 'seq2seq':
            self.transformer = Seq2Seq(vocab_size, **transformer_args)
        elif seq_modeling == 'convseq2seq':
            self.transformer = ConvSeq2Seq(vocab_size, **transformer_args)
        else:
            raise 'Not Support Seq Model'

    def forward(self, img, tgt_input, tgt_key_padding_mask):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        src = self.cnn(img)
        if self.seq_modeling == 'transformer':
            outputs = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        elif self.seq_modeling == 'seq2seq':
            outputs = self.transformer(src, tgt_input)
        elif self.seq_modeling == 'convseq2seq':
            outputs = self.transformer(src, tgt_input)
        return outputs

class Beam:

    def __init__(self, beam_size=8, min_length=0, n_top=1, ranker=None, start_token_id=1, end_token_id=2):
        self.beam_size = beam_size
        self.min_length = min_length
        self.ranker = ranker
        self.end_token_id = end_token_id
        self.top_sentence_ended = False
        self.prev_ks = []
        self.next_ys = [torch.LongTensor(beam_size).fill_(start_token_id)]
        self.current_scores = torch.FloatTensor(beam_size).zero_()
        self.all_scores = []
        self.finished = []
        self.n_top = n_top
        self.ranker = ranker

    def advance(self, next_log_probs):
        vocabulary_size = next_log_probs.size(1)
        current_length = len(self.next_ys)
        if current_length < self.min_length:
            for beam_index in range(len(next_log_probs)):
                next_log_probs[beam_index][self.end_token_id] = -10000000000.0
        if len(self.prev_ks) > 0:
            beam_scores = next_log_probs + self.current_scores.unsqueeze(1).expand_as(next_log_probs)
            last_y = self.next_ys[-1]
            for beam_index in range(last_y.size(0)):
                if last_y[beam_index] == self.end_token_id:
                    beam_scores[beam_index] = -10000000000.0
        else:
            beam_scores = next_log_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        top_scores, top_score_ids = flat_beam_scores.topk(k=self.beam_size, dim=0, largest=True, sorted=True)
        self.current_scores = top_scores
        self.all_scores.append(self.current_scores)
        prev_k = top_score_ids // vocabulary_size
        next_y = top_score_ids - prev_k * vocabulary_size
        self.prev_ks.append(prev_k)
        self.next_ys.append(next_y)
        for beam_index, last_token_id in enumerate(next_y):
            if last_token_id == self.end_token_id:
                self.finished.append((self.current_scores[beam_index], len(self.next_ys) - 1, beam_index))
        if next_y[0] == self.end_token_id:
            self.top_sentence_ended = True

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return torch.stack(self.next_ys, dim=1)

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def done(self):
        return self.top_sentence_ended and len(self.finished) >= self.n_top

    def get_hypothesis(self, timestep, k):
        hypothesis = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hypothesis.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return hypothesis[::-1]

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            while len(self.finished) < minimum:
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1
        self.finished = sorted(self.finished, key=lambda a: a[0], reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return (scores, ks)

class Vocab:

    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3
        self.chars = chars
        self.c2i = {c: i + 4 for i, c in enumerate(chars)}
        self.i2c = {i + 4: c for i, c in enumerate(chars)}
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]

    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent

    def __len__(self):
        return len(self.c2i) + 4

    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars

def translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    model.eval()
    device = img.device
    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)
        sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)
    return sent

def beamsearch(memory, model, device, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    model.eval()
    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token, end_token_id=eos_token)
    with torch.no_grad():
        memory = model.transformer.expand_memory(memory, beam_size)
        for _ in range(max_seq_length):
            tgt_inp = beam.get_current_state().transpose(0, 1).to(device)
            decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)
            log_prob = log_softmax(decoder_outputs[:, -1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())
            if beam.done():
                break
        scores, ks = beam.sort_finished(minimum=1)
        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
    return [1] + [int(i) for i in hypothesises[0][:-1]]

def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCXHxW"""
    model.eval()
    device = img.device
    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)
        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1] * len(img)]
        max_length = 0
        while max_length <= max_seq_length and (not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1))):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to('cpu')
            values, indices = torch.topk(output, 5)
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)
            translated_sentence.append(indices)
            max_length += 1
            del output
        translated_sentence = np.asarray(translated_sentence).T
        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)
        char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)
    return (translated_sentence, char_probs)

def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    model = VietOCR(len(vocab), config['backbone'], config['cnn'], config['transformer'], config['seq_modeling'])
    model = model.to(device)
    return (model, vocab)

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)
    return (new_w, expected_height)

def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert('RGB')
    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)
    img = img.resize((new_w, image_height), Image.Resampling.LANCZOS)
    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img

def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img
