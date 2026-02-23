"""
Neural Network Architectures for Multi-Timeframe SAC Agent
- Custom GRU/LSTM cells (from scratch, no nn.GRU/nn.LSTM)
- Multi-timeframe encoders
- Cross-timeframe attention
- GPU compatible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


# ============================================================
# Custom RNN Cells (from scratch)
# ============================================================

class CustomGRUCell(nn.Module):
    """
    Custom GRU cell implemented from scratch.
    
    GRU equations:
        z = sigmoid(W_z * [h, x] + b_z)  # update gate
        r = sigmoid(W_r * [h, x] + b_r)  # reset gate
        h_tilde = tanh(W_h * [r*h, x] + b_h)  # candidate hidden
        h_new = (1 - z) * h + z * h_tilde
    
    This is GPU compatible and avoids nn.GRU issues.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Combined linear layer for efficiency: [h, x] -> z, r, h_tilde
        self.W_zrh = nn.Linear(input_dim + hidden_dim, 3 * hidden_dim)
        
        # Separate layer for candidate (with reset gate applied)
        self.W_candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
            h: (batch, hidden_dim)
        Returns:
            h_new: (batch, hidden_dim)
        """
        # Concatenate hidden and input
        combined = torch.cat([h, x], dim=-1)
        
        # Compute update (z) and reset (r) gates
        gates = self.W_zrh(combined)
        z, r, _ = gates.chunk(3, dim=-1)
        
        z = torch.sigmoid(z)  # Update gate
        r = torch.sigmoid(r)  # Reset gate
        
        # Candidate hidden state with reset gate
        reset_combined = torch.cat([r * h, x], dim=-1)
        h_tilde = torch.tanh(self.W_candidate(reset_combined))
        
        # New hidden state
        h_new = (1.0 - z) * h + z * h_tilde
        
        return h_new
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)


class CustomLSTMCell(nn.Module):
    """
    Custom LSTM cell implemented from scratch.
    
    LSTM equations:
        f = sigmoid(W_f * [h, x] + b_f)  # forget gate
        i = sigmoid(W_i * [h, x] + b_i)  # input gate
        o = sigmoid(W_o * [h, x] + b_o)  # output gate
        c_tilde = tanh(W_c * [h, x] + b_c)  # candidate cell
        c_new = f * c + i * c_tilde  # new cell state
        h_new = o * tanh(c_new)  # new hidden state
    
    This is GPU compatible and avoids nn.LSTM issues.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Combined linear layer for all gates: [h, x] -> f, i, o, c_tilde
        self.W_gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)
            state: (h, c) where h, c are (batch, hidden_dim)
        Returns:
            (h_new, c_new): New hidden and cell states
        """
        h, c = state
        
        # Concatenate hidden and input
        combined = torch.cat([h, x], dim=-1)
        
        # Compute all gates in one pass
        gates = self.W_gates(combined)
        f, i, o, c_tilde = gates.chunk(4, dim=-1)
        
        # Apply activations
        f = torch.sigmoid(f)       # Forget gate
        i = torch.sigmoid(i)       # Input gate
        o = torch.sigmoid(o)       # Output gate
        c_tilde = torch.tanh(c_tilde)  # Candidate cell
        
        # New states
        c_new = f * c + i * c_tilde
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states to zeros."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c


class CustomGRULayer(nn.Module):
    """
    Multi-layer GRU using custom cells.
    Processes sequence step by step.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Create GRU cells for each layer
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(CustomGRUCell(in_dim, hidden_dim))
    
    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence through GRU layers.
        
        Args:
            x: (batch, seq_len, input_dim)
            h0: (num_layers, batch, hidden_dim) initial hidden state
            
        Returns:
            output: (batch, hidden_dim) final hidden state
            h_n: (num_layers, batch, hidden_dim) all final hidden states
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize hidden states
        if h0 is None:
            h = [cell.init_hidden(batch_size, device) for cell in self.cells]
        else:
            h = [h0[layer] for layer in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer, cell in enumerate(self.cells):
                h[layer] = cell(x_t, h[layer])
                x_t = h[layer]
        
        # Stack hidden states
        h_n = torch.stack(h, dim=0)
        
        return h[-1], h_n


class CustomLSTMLayer(nn.Module):
    """
    Multi-layer LSTM using custom cells.
    Processes sequence step by step.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Create LSTM cells for each layer
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(CustomLSTMCell(in_dim, hidden_dim))
    
    def forward(self, x: torch.Tensor, h0: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process sequence through LSTM layers.
        
        Args:
            x: (batch, seq_len, input_dim)
            h0: ((num_layers, batch, hidden_dim), (num_layers, batch, hidden_dim)) initial states
            
        Returns:
            output: (batch, hidden_dim) final hidden state
            (h_n, c_n): Final hidden and cell states
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize states
        if h0 is None:
            h = [cell.init_hidden(batch_size, device) for cell in self.cells]
        else:
            h0_tensor, c0_tensor = h0
            h = [(h0_tensor[layer], c0_tensor[layer]) for layer in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer, cell in enumerate(self.cells):
                h[layer] = cell(x_t, h[layer])
                x_t = h[layer][0]  # Pass hidden state to next layer
        
        # Stack states
        h_n = torch.stack([state[0] for state in h], dim=0)
        c_n = torch.stack([state[1] for state in h], dim=0)
        
        return h[-1][0], (h_n, c_n)


# ============================================================
# Multi-Timeframe Encoder
# ============================================================

class TimeframeEncoder(nn.Module):
    """
    Encoder for a single timeframe.
    Uses custom RNN to process sequence and produce encoding.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 rnn_type: str = "gru", dropout: float = 0.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        
        # Select RNN type
        if rnn_type == "lstm":
            self.rnn = CustomLSTMLayer(input_dim, hidden_dim, num_layers)
        else:  # Default to GRU
            self.rnn = CustomGRULayer(input_dim, hidden_dim, num_layers)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode timeframe sequence.
        
        Args:
            x: (batch, seq_len, input_dim)
            
        Returns:
            encoding: (batch, hidden_dim)
        """
        # Handle NaN/Inf
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.where(torch.isinf(x), torch.sign(x) * 10.0, x)
        
        # Process through RNN
        encoding, _ = self.rnn(x)
        
        # Normalize and dropout
        encoding = self.layer_norm(encoding)
        encoding = self.dropout(encoding)
        
        return encoding


class MultiTimeframeEncoder(nn.Module):
    """
    Multi-timeframe encoder with hierarchical processing.
    
    Architecture:
    - Separate encoder for each timeframe
    - Cross-timeframe attention to combine
    - Output: combined context vector
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.timeframes = config.get('timeframes', ['M1', 'M5', 'M15', 'M30', 'H1'])
        self.encoder_dims = config.get('encoder_dims', {})
        self.feature_dims = config.get('feature_dims', {})  # Input dims per TF
        self.rnn_type = config.get('rnn_type', 'gru')
        self.dropout = config.get('dropout', 0.1)
        
        # Create encoder for each timeframe
        self.encoders = nn.ModuleDict()
        
        for tf in self.timeframes:
            tf_config = self.encoder_dims.get(tf, {'hidden': 64, 'layers': 1})
            input_dim = self.feature_dims.get(tf, 30)  # Default feature count
            hidden_dim = tf_config['hidden']
            num_layers = tf_config['layers']
            
            self.encoders[tf] = TimeframeEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                rnn_type=self.rnn_type,
                dropout=self.dropout
            )
        
        # Calculate total encoding dimension
        self.total_encoding_dim = sum(
            self.encoder_dims.get(tf, {'hidden': 64})['hidden']
            for tf in self.timeframes
        )
        
        # Cross-timeframe attention
        attention_dim = config.get('attention_dim', 256)
        self.attention = CrossTimeframeAttention(
            encoding_dims={tf: self.encoder_dims.get(tf, {'hidden': 64})['hidden'] 
                          for tf in self.timeframes},
            attention_dim=attention_dim,
            num_heads=config.get('attention_heads', 4)
        )
        
        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.total_encoding_dim + attention_dim, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, mtf_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode all timeframes and combine.
        
        Args:
            mtf_input: Dict mapping timeframe to (batch, seq_len, features)
            
        Returns:
            context: (batch, attention_dim) combined context
        """
        # Encode each timeframe
        encodings = {}
        for tf in self.timeframes:
            if tf in mtf_input:
                encodings[tf] = self.encoders[tf](mtf_input[tf])
            else:
                # Zero encoding if missing
                batch_size = next(iter(mtf_input.values())).size(0)
                hidden = self.encoder_dims.get(tf, {'hidden': 64})['hidden']
                encodings[tf] = torch.zeros(batch_size, hidden, 
                                           device=next(iter(mtf_input.values())).device)
        
        # Apply cross-timeframe attention
        attention_out = self.attention(encodings)
        
        # Concatenate all encodings
        all_encodings = torch.cat([encodings[tf] for tf in self.timeframes], dim=-1)
        
        # Combine with attention output
        combined = torch.cat([all_encodings, attention_out], dim=-1)
        
        # Project to output
        context = self.output_proj(combined)
        
        return context


class CrossTimeframeAttention(nn.Module):
    """
    Cross-timeframe attention mechanism.
    
    Uses M1 encoding as query, other timeframes as keys/values.
    Allows the model to focus on relevant HTF context.
    """
    
    def __init__(self, encoding_dims: Dict[str, int], attention_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.encoding_dims = encoding_dims
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        
        # Query projection (from M1)
        self.query_proj = nn.Linear(encoding_dims.get('M1', 128), attention_dim)
        
        # Key/Value projections (from all timeframes)
        total_kv_dim = sum(encoding_dims.values())
        self.key_proj = nn.Linear(total_kv_dim, attention_dim)
        self.value_proj = nn.Linear(total_kv_dim, attention_dim)
        
        # Output projection
        self.out_proj = nn.Linear(attention_dim, attention_dim)
        
        # Scaling factor
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, encodings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply cross-timeframe attention.
        
        Args:
            encodings: Dict mapping timeframe to (batch, hidden_dim)
            
        Returns:
            output: (batch, attention_dim)
        """
        batch_size = next(iter(encodings.values())).size(0)
        device = next(iter(encodings.values())).device
        
        # Query from M1
        query = self.query_proj(encodings['M1'])  # (batch, attention_dim)
        
        # Keys and values from all timeframes
        all_encodings = torch.cat([encodings[tf] for tf in sorted(encodings.keys())], dim=-1)
        key = self.key_proj(all_encodings)    # (batch, attention_dim)
        value = self.value_proj(all_encodings)  # (batch, attention_dim)
        
        # Reshape for multi-head attention
        # (batch, attention_dim) -> (batch, num_heads, head_dim)
        query = query.view(batch_size, self.num_heads, self.head_dim)
        key = key.view(batch_size, self.num_heads, self.head_dim)
        value = value.view(batch_size, self.num_heads, self.head_dim)
        
        # Attention scores: (batch, num_heads, 1)
        # For single query, single key (not seq2seq)
        scores = (query * key).sum(dim=-1, keepdim=True) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_out = attn_weights * value  # (batch, num_heads, head_dim)
        attn_out = attn_out.view(batch_size, self.attention_dim)
        
        # Output projection
        output = self.out_proj(attn_out)
        
        return output


# ============================================================
# Actor & Critic Networks
# ============================================================

class ResBlock(nn.Module):
    """Pre-norm residual MLP block."""
    
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class MTFActorNetwork(nn.Module):
    """
    Multi-timeframe Actor network for SAC.
    
    Input: Multi-timeframe observations
    Output: Action in [-1, 1] and log-prob
    """
    
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0
    
    def __init__(self, mtf_config: dict, action_dim: int = 1):
        super().__init__()
        
        # Multi-timeframe encoder
        self.mtf_encoder = MultiTimeframeEncoder(mtf_config)
        
        # Portfolio state processing
        portfolio_dim = 6  # cash_ratio, position_value, abs_position, unrealized_pnl_pct, div_bull, div_bear
        self.portfolio_proj = nn.Sequential(
            nn.Linear(portfolio_dim, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )
        
        # Combined dimension
        combined_dim = mtf_config.get('attention_dim', 256) + 32
        
        # Policy trunk
        hidden_dim = mtf_config.get('hidden_dim', 256)
        n_layers = mtf_config.get('n_layers', 3)
        dropout = mtf_config.get('dropout', 0.1)
        
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(n_layers)
        ])
        
        # Policy heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, mtf_obs: Dict[str, torch.Tensor], 
                portfolio: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            mtf_obs: Dict mapping timeframe to (batch, seq_len, features)
            portfolio: (batch, 5) portfolio state
            deterministic: Whether to sample deterministically
            
        Returns:
            action: (batch, action_dim)
            log_prob: (batch, 1) or None if deterministic
        """
        # Encode multi-timeframe observations
        mtf_encoding = self.mtf_encoder(mtf_obs)
        
        # Process portfolio state
        portfolio_encoding = self.portfolio_proj(portfolio)
        
        # Combine
        combined = torch.cat([mtf_encoding, portfolio_encoding], dim=-1)
        
        # Policy trunk
        x = self.trunk(combined)
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy heads
        mean = self.mean_head(x)
        
        if deterministic:
            action = torch.tanh(mean)
            return action, None
        
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        
        # Reparameterization trick
        eps = torch.randn_like(mean)
        x_t = mean + std * eps
        action = torch.tanh(x_t)
        
        # Log-prob of tanh-squashed Gaussian
        log_prob = (
            -0.5 * (eps ** 2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
            - torch.log(1.0 - action ** 2 + 1e-6)
        ).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, mtf_obs: Dict[str, torch.Tensor],
                   portfolio: torch.Tensor,
                   deterministic: bool = False) -> torch.Tensor:
        """Get action without gradient."""
        with torch.no_grad():
            action, _ = self.forward(mtf_obs, portfolio, deterministic)
        return action


class MTFCriticNetwork(nn.Module):
    """
    Multi-timeframe Critic network for SAC.
    
    Twin Q-networks with shared encoder.
    """
    
    def __init__(self, mtf_config: dict, action_dim: int = 1):
        super().__init__()
        
        # Multi-timeframe encoder (shared)
        self.mtf_encoder = MultiTimeframeEncoder(mtf_config)
        
        # Portfolio state processing
        portfolio_dim = 6  # cash_ratio, position_value, abs_position, unrealized_pnl_pct, div_bull, div_bear
        self.portfolio_proj = nn.Sequential(
            nn.Linear(portfolio_dim, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )
        
        # Combined dimension
        combined_dim = mtf_config.get('attention_dim', 256) + 32 + action_dim
        
        # Q-network trunk
        hidden_dim = mtf_config.get('hidden_dim', 256)
        n_layers = mtf_config.get('n_layers', 3)
        dropout = mtf_config.get('dropout', 0.1)
        
        # Twin Q-networks
        self.q1_trunk = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.q2_trunk = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks for each Q
        self.q1_res = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.q2_res = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        
        # Q heads
        self.q1_head = nn.Linear(hidden_dim, 1)
        self.q2_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, mtf_obs: Dict[str, torch.Tensor],
                portfolio: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            mtf_obs: Dict mapping timeframe to (batch, seq_len, features)
            portfolio: (batch, 5) portfolio state
            action: (batch, action_dim)
            
        Returns:
            q1, q2: (batch, 1) each
        """
        # Encode multi-timeframe observations
        mtf_encoding = self.mtf_encoder(mtf_obs)
        
        # Process portfolio state
        portfolio_encoding = self.portfolio_proj(portfolio)
        
        # Combine with action
        combined = torch.cat([mtf_encoding, portfolio_encoding, action], dim=-1)
        
        # Q1
        q1 = self.q1_trunk(combined)
        for res_block in self.q1_res:
            q1 = res_block(q1)
        q1 = self.q1_head(q1)
        
        # Q2
        q2 = self.q2_trunk(combined)
        for res_block in self.q2_res:
            q2 = res_block(q2)
        q2 = self.q2_head(q2)
        
        return q1, q2


# ============================================================
# Legacy Networks (for backward compatibility)
# ============================================================

class ActorNetwork(nn.Module):
    """Legacy single-timeframe actor for backward compatibility."""
    
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 384,
                 n_heads: int = 4,
                 gru_hidden: int = 128,
                 n_layers: int = 3,
                 use_temporal: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        
        self.use_temporal = use_temporal
        self.hidden_dim = hidden_dim
        
        if use_temporal:
            self.temporal = CustomGRULayer(state_dim, gru_hidden, num_layers=1)
            trunk_input_dim = gru_hidden
        else:
            self.temporal = None
            trunk_input_dim = state_dim
        
        neck = max(gru_hidden, hidden_dim // 2)
        
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        
        self.neck = nn.Sequential(
            nn.Linear(hidden_dim, neck),
            nn.LayerNorm(neck),
            nn.GELU(),
        )
        
        self.mean_head = nn.Linear(neck, action_dim)
        self.log_std_head = nn.Linear(neck, action_dim)
    
    def forward(self, state: torch.Tensor, deterministic: bool = False):
        state = state.clamp(-1e6, 1e6)
        
        if self.use_temporal and self.temporal is not None:
            if state.dim() == 2:
                state = state.unsqueeze(1)
            state, _ = self.temporal(state)
        
        x = self.trunk(state)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.neck(x)
        
        mean = self.mean_head(x)
        
        if deterministic:
            action = torch.tanh(mean)
            return action, None
        
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        
        eps = torch.randn_like(mean)
        x_t = mean + std * eps
        action = torch.tanh(x_t)
        
        log_prob = (
            -0.5 * ((eps) ** 2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
            - torch.log(1.0 - action ** 2 + 1e-6)
        ).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            action, _ = self.forward(state, deterministic)
        return action


class CriticNetwork(nn.Module):
    """Legacy single-timeframe critic for backward compatibility."""
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 384,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        in_dim = state_dim + action_dim
        
        self.q1_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.q1_res = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.q1_head = nn.Linear(hidden_dim, 1)
        
        self.q2_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.q2_res = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.q2_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        
        q1 = self.q1_net(sa)
        for res in self.q1_res:
            q1 = res(q1)
        q1 = self.q1_head(q1)
        
        q2 = self.q2_net(sa)
        for res in self.q2_res:
            q2 = res(q2)
        q2 = self.q2_head(q2)
        
        return q1, q2


# ============================================================
# Helper Functions
# ============================================================

def init_weights(module: nn.Module) -> None:
    """Initialize weights with Xavier uniform."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_mtf_networks(config: dict) -> Tuple[MTFActorNetwork, MTFCriticNetwork]:
    """
    Build multi-timeframe actor and critic networks.
    
    Args:
        config: Configuration dictionary containing MTF and network settings
        
    Returns:
        actor, critic: MTF networks
    """
    actor = MTFActorNetwork(config, action_dim=1)
    critic = MTFCriticNetwork(config, action_dim=1)
    
    actor.apply(init_weights)
    critic.apply(init_weights)
    
    return actor, critic


def build_networks(state_dim: int, action_dim: int, config: dict):
    """Legacy function - build single-timeframe networks."""
    hidden_dim = config.get('hidden_dim', 384)
    n_heads = config.get('attention_heads', 4)
    gru_hidden = config.get('gru_hidden', 128)
    n_layers = config.get('n_layers', 3)
    use_temporal = config.get('use_temporal', False)
    dropout = config.get('dropout', 0.1)
    
    actor = ActorNetwork(
        state_dim, action_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        gru_hidden=gru_hidden,
        n_layers=n_layers,
        use_temporal=use_temporal,
        dropout=dropout
    )
    
    critic = CriticNetwork(
        state_dim, action_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    )
    
    actor.apply(init_weights)
    critic.apply(init_weights)
    
    return actor, critic