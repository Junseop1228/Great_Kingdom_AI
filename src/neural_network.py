# src/neural_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    ResNet의 기본 구성 요소인 잔차 블록입니다.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # 스킵 연결 (Skip Connection)
        out = F.relu(out)
        return out

class GreatKingdomNet(nn.Module):
    """
    '그레이트 킹덤'을 위한 딥러닝 신경망 (경량화 버전)
    """
    # 잔차 블록 5개, 채널 64개로 설정하여 빠른 훈련 속도에 초점
    def __init__(self, num_res_blocks=5, num_channels=64):
        super().__init__()
        # 초기 컨볼루션 레이어
        self.conv_in = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        # 잔차 블록 몸통
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        
        # 정책 머리 (Policy Head)
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 9 * 9, 81 + 1)

        # 가치 머리 (Value Head)
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 9 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 몸통
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)
        
        # 정책 머리
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(-1, 2 * 9 * 9)
        policy = self.policy_fc(policy)
        
        # 가치 머리
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(-1, 1 * 9 * 9)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        # 최종적으로 정책에 대한 로그 소프트맥스와 가치를 반환
        return F.log_softmax(policy, dim=1), value
