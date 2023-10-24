import torch.nn as nn
from models import CNN6, AudioCNN
import torch

class MultiModal(nn.Module):
    def __init__(self, d_model=4, nhead=8, num_classes=4, mel_dim=4):
        super(MultiModal, self).__init__()

        # Audio processing with Transformer
        self.transformer = AudioCNN()

        # Assuming CNN10 is still used for mel processing
        # Placeholder for CNN10 as its definition wasn't provided in this context
        self.cnn6 = CNN6()

        self.attention = MultiModalAttention(audio_dim=d_model, mel_dim=mel_dim)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, audio_input, mel_input):
        audio_features = self.transformer(audio_input)
        # print("audio_input shape: ", audio_features.shape)
        mel_features = self.cnn6(mel_input)
        # print("mel_input shape: ", mel_features.shape)
        # Using attention to combine features
        attention_output = self.attention(audio_features, mel_features)

        # Classifier for final output
        out = self.classifier(attention_output)
        return out


class MultiModalAttention(nn.Module):
    def __init__(self, audio_dim=4, mel_dim=4):
        super(MultiModalAttention, self).__init__()
        self.query = nn.Linear(audio_dim, mel_dim)
        self.key = nn.Linear(mel_dim, mel_dim)
        self.value = nn.Linear(mel_dim, audio_dim)

    def forward(self, audio_features, mel_features):
        Q = self.query(audio_features)
        K = self.key(mel_features)
        V = self.value(mel_features)

        # Compute attention scores
        attention_scores = torch.einsum("bn,bn->bn", [Q, K]).softmax(dim=1)

        # Compute weighted features
        weighted_audio_features = attention_scores * audio_features
        weighted_mel_features = (1 - attention_scores) * mel_features

        # Sum up the weighted features for majority voting
        out = weighted_audio_features + weighted_mel_features

        return out
