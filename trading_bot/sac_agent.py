"""
Soft Actor-Critic (SAC) Agent Implementation
- Multi-timeframe support
- Pure PyTorch implementation (no stable-baselines)
- Automatic entropy tuning
- CUDA GPU acceleration with mixed precision (AMP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import os
from tqdm import tqdm

from .networks import (
    ActorNetwork, CriticNetwork, build_networks,
    MTFActorNetwork, MTFCriticNetwork, build_mtf_networks
)
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, create_buffer
from .config import MTF_CONFIG, SAC_CONFIG

# Enable cudnn autotuner for consistent-size inputs (big speedup for RNNs)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True


class MTFReplayBuffer:
    """
    Replay buffer for multi-timeframe observations.
    Stores observations as dictionaries keyed by timeframe.
    """
    
    def __init__(self, capacity: int, timeframes: List[str], feature_dims: Dict[str, int],
                 bars_per_tf: Dict[str, int], portfolio_dim: int = 5, action_dim: int = 1):
        self.capacity = capacity
        self.timeframes = timeframes
        self.feature_dims = feature_dims
        self.bars_per_tf = bars_per_tf
        self.portfolio_dim = portfolio_dim
        self.action_dim = action_dim
        
        # Storage for each timeframe
        self.mtf_obs = {tf: [] for tf in timeframes}
        self.mtf_next_obs = {tf: [] for tf in timeframes}
        self.portfolios = []
        self.next_portfolios = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        self.position = 0
        self.size = 0
        
    def push(self, mtf_obs: Dict[str, np.ndarray], portfolio: np.ndarray,
             action: np.ndarray, reward: float, 
             next_mtf_obs: Dict[str, np.ndarray], next_portfolio: np.ndarray,
             done: bool):
        """Store a transition."""
        
        # Expand capacity if needed
        if len(self.actions) < self.capacity:
            for tf in self.timeframes:
                self.mtf_obs[tf].append(None)
                self.mtf_next_obs[tf].append(None)
            self.portfolios.append(None)
            self.next_portfolios.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.dones.append(None)
        
        # Store at current position
        for tf in self.timeframes:
            self.mtf_obs[tf][self.position] = mtf_obs.get(tf, np.zeros((self.bars_per_tf[tf], self.feature_dims[tf]), dtype=np.float32))
            self.mtf_next_obs[tf][self.position] = next_mtf_obs.get(tf, np.zeros((self.bars_per_tf[tf], self.feature_dims[tf]), dtype=np.float32))
        
        self.portfolios[self.position] = portfolio
        self.next_portfolios[self.position] = next_portfolio
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, batch_size)
        
        mtf_obs_batch = {
            tf: torch.FloatTensor(np.stack([self.mtf_obs[tf][i] for i in indices]))
            for tf in self.timeframes
        }
        mtf_next_obs_batch = {
            tf: torch.FloatTensor(np.stack([self.mtf_next_obs[tf][i] for i in indices]))
            for tf in self.timeframes
        }
        
        portfolio_batch = torch.FloatTensor(np.stack([self.portfolios[i] for i in indices]))
        next_portfolio_batch = torch.FloatTensor(np.stack([self.next_portfolios[i] for i in indices]))
        actions_batch = torch.FloatTensor(np.stack([self.actions[i] for i in indices]))
        # Ensure proper shapes: rewards and dones should be (batch, 1)
        rewards_batch = torch.FloatTensor(np.array([self.rewards[i] for i in indices])).unsqueeze(1)
        dones_batch = torch.FloatTensor(np.array([self.dones[i] for i in indices])).unsqueeze(1)
        
        return (mtf_obs_batch, portfolio_batch, actions_batch, 
                rewards_batch, mtf_next_obs_batch, next_portfolio_batch, dones_batch)
    
    def __len__(self):
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


class SACAgentMTF:
    """
    Multi-Timeframe SAC Agent
    
    Key features:
    - Processes observations from multiple timeframes
    - Uses MTFActorNetwork and MTFCriticNetwork
    - Automatic entropy coefficient tuning
    - CUDA GPU acceleration with mixed precision (AMP)
    """
    
    def __init__(
        self,
        mtf_config: dict,
        action_dim: int = 1,
        device: str = 'auto',
    ):
        self.mtf_config = mtf_config
        self.action_dim = action_dim
        
        # Device setup
        if device == 'auto':
            self.device = self._get_device()
        else:
            self.device = device
        
        self.use_cuda = self.device.startswith('cuda')
        print(f"[SAC-MTF] Using device: {self.device}")
        
        # Mixed precision (AMP) — safe for RL, big speedup on GPU
        self.use_amp = self.use_cuda
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print("[SAC-MTF] Mixed precision (AMP) enabled")
        
        # Build MTF networks
        self.actor, self.critic = build_mtf_networks(mtf_config)
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        # Target networks
        _, self.critic_target = build_mtf_networks(mtf_config)
        self.critic_target.to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Freeze target networks
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # torch.compile for PyTorch 2.x (big speedup for forward/backward)
        if hasattr(torch, 'compile') and self.use_cuda:
            try:
                self.actor = torch.compile(self.actor)
                self.critic = torch.compile(self.critic)
                self.critic_target = torch.compile(self.critic_target)
                print("[SAC-MTF] torch.compile enabled")
            except Exception as e:
                print(f"[SAC-MTF] torch.compile not available: {e}")
        
        # Entropy coefficient
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=mtf_config.get('actor_lr', mtf_config.get('learning_rate', 3e-4))
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=mtf_config.get('critic_lr', mtf_config.get('learning_rate', 3e-4))
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=mtf_config.get('alpha_lr', 3e-4)
        )
        
        # Replay buffer
        self.timeframes = mtf_config.get('timeframes', ['M1', 'M5', 'M15', 'M30', 'H1'])
        self.bars_per_tf = mtf_config.get('bars_per_tf', {'M1': 360, 'M5': 72, 'M15': 24, 'M30': 12, 'H1': 6})
        self.feature_dims = mtf_config.get('feature_dims', {tf: 30 for tf in self.timeframes})
        
        self.buffer = MTFReplayBuffer(
            capacity=mtf_config.get('buffer_size', 500_000),
            timeframes=self.timeframes,
            feature_dims=self.feature_dims,
            bars_per_tf=self.bars_per_tf,
            portfolio_dim=5,
            action_dim=action_dim
        )
        
        # Training parameters
        self.gamma = mtf_config.get('gamma', 0.99)
        self.tau = mtf_config.get('tau', 0.005)
        self.batch_size = mtf_config.get('batch_size', 128)
        self.learning_starts = mtf_config.get('learning_starts', 1000)
        self.gradient_steps = mtf_config.get('gradient_steps', 1)
        
        # Training state
        self.total_steps = 0
        self.training_step = 0
        
    def _get_device(self) -> str:
        """Auto-detect best available device (CUDA > CPU)."""
        if torch.cuda.is_available():
            print(f"[SAC-MTF] CUDA available: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        print("[SAC-MTF] Using CPU (no GPU detected)")
        return 'cpu'
    
    def _obs_to_device(self, mtf_obs: Dict[str, np.ndarray], portfolio: np.ndarray) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Convert numpy observations to device tensors."""
        mtf_tensor = {
            tf: torch.FloatTensor(mtf_obs[tf]).unsqueeze(0).to(self.device, non_blocking=True)
            for tf in self.timeframes
        }
        portfolio_tensor = torch.FloatTensor(portfolio).unsqueeze(0).to(self.device, non_blocking=True)
        return mtf_tensor, portfolio_tensor
    
    def select_action(self, mtf_obs: Dict[str, np.ndarray], portfolio: np.ndarray) -> np.ndarray:
        """Select action with exploration noise."""
        with torch.no_grad():
            mtf_tensor, portfolio_tensor = self._obs_to_device(mtf_obs, portfolio)
            action, _ = self.actor(mtf_tensor, portfolio_tensor, deterministic=False)
        return action.cpu().numpy()[0]
    
    def select_action_deterministic(self, mtf_obs: Dict[str, np.ndarray], portfolio: np.ndarray) -> np.ndarray:
        """Select action without exploration noise."""
        with torch.no_grad():
            mtf_tensor, portfolio_tensor = self._obs_to_device(mtf_obs, portfolio)
            action, _ = self.actor(mtf_tensor, portfolio_tensor, deterministic=True)
        return action.cpu().numpy()[0]
    
    def store_transition(
        self,
        mtf_obs: Dict[str, np.ndarray],
        portfolio: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_mtf_obs: Dict[str, np.ndarray],
        next_portfolio: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.buffer.push(mtf_obs, portfolio, action, reward, next_mtf_obs, next_portfolio, done)
        self.total_steps += 1
    
    def update(self) -> Optional[Dict[str, float]]:
        """Perform one training step with AMP mixed precision."""
        if not self.buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        (mtf_obs, portfolio, actions, rewards, 
         next_mtf_obs, next_portfolio, dones) = self.buffer.sample(self.batch_size)
        
        # Move to device with non-blocking transfers
        mtf_obs = {tf: mtf_obs[tf].to(self.device, non_blocking=True) for tf in self.timeframes}
        next_mtf_obs = {tf: next_mtf_obs[tf].to(self.device, non_blocking=True) for tf in self.timeframes}
        portfolio = portfolio.to(self.device, non_blocking=True)
        next_portfolio = next_portfolio.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True)
        dones = dones.to(self.device, non_blocking=True)
        
        amp_ctx = torch.amp.autocast('cuda', enabled=self.use_amp)
        
        # ==================== CRITIC UPDATE ====================
        with torch.no_grad():
            with amp_ctx:
                next_actions, next_log_probs = self.actor(next_mtf_obs, next_portfolio, deterministic=False)
                q1_target, q2_target = self.critic_target(next_mtf_obs, next_portfolio, next_actions)
                q_target_min = torch.min(q1_target, q2_target)
                q_target = rewards + (1 - dones) * self.gamma * (
                    q_target_min - self.alpha * next_log_probs
                )
        
        with amp_ctx:
            q1_current, q2_current = self.critic(mtf_obs, portfolio, actions)
            q_target = torch.clamp(q_target, -100.0, 100.0)
            critic1_loss = F.smooth_l1_loss(q1_current, q_target)
            critic2_loss = F.smooth_l1_loss(q2_current, q_target)
            critic_loss = critic1_loss + critic2_loss
        
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.scaler.step(self.critic_optimizer)
        else:
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # ==================== ACTOR UPDATE ====================
        with amp_ctx:
            new_actions, log_probs = self.actor(mtf_obs, portfolio, deterministic=False)
            log_probs = torch.clamp(log_probs, -10, 10)
            q1_new, q2_new = self.critic(mtf_obs, portfolio, new_actions)
            q_new_min = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_probs - q_new_min).mean()
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.scaler.step(self.actor_optimizer)
        else:
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
        
        # ==================== ALPHA UPDATE ====================
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(alpha_loss).backward()
            self.scaler.unscale_(self.alpha_optimizer)
            torch.nn.utils.clip_grad_norm_([self.log_alpha], 0.5)
            self.scaler.step(self.alpha_optimizer)
            self.scaler.update()
        else:
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], 0.5)
            self.alpha_optimizer.step()
        
        self.log_alpha.data = torch.clamp(self.log_alpha.data, -5, 2)
        self.alpha = self.log_alpha.exp().item()
        
        # ==================== SOFT TARGET UPDATE ====================
        self._soft_update(self.critic, self.critic_target, self.tau)
        
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
            'q1_mean': q1_current.mean().item(),
            'q2_mean': q2_current.mean().item(),
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        """Polyak averaging for target network update."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )
    
    def learn(
        self,
        total_steps: int,
        env,
        callback=None,
        checkpoint_dir: str = 'models',
        checkpoint_interval: int = 50_000,
        best_model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop for MTF environment.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        if best_model_path is None:
            best_model_path = os.path.join(checkpoint_dir, 'sac_mtf_best.pt')
        
        obs, _ = env.reset()
        mtf_obs = obs  # Dict with timeframe keys
        portfolio = obs.get('portfolio', np.zeros(5, dtype=np.float32))
        
        episode = 0
        episode_reward = 0.0
        episode_length = 0
        episode_trades = 0
        episode_rewards = []
        episode_trade_list = []
        
        best_episode_reward = -float('inf')
        
        metrics_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_trades': [],
            'critic_loss': [],
            'actor_loss': [],
            'alpha': [],
        }
        
        pbar = tqdm(
            range(total_steps), desc="Training MTF", unit="step",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
        )
        
        last_metrics = {}
        last_checkpoint_step = 0
        
        print(
            "\n{:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}  {:>7}  {:>7}  {}".format(
                "Ep", "Step", "Reward", "Avg10", "Best", "Len",
                "Trades", "Alpha", "CriticL",
            )
        )
        print("-" * 75)
        
        for step in pbar:
            # Action selection
            if step < self.learning_starts:
                action = env.action_space.sample()
            else:
                action = self.select_action(mtf_obs, portfolio)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_mtf_obs = next_obs
            next_portfolio = next_obs.get('portfolio', np.zeros(5, dtype=np.float32))
            
            ep_trade_count = info.get('trade_count', 0)
            
            # Store transition
            self.store_transition(
                mtf_obs, portfolio, action, reward,
                next_mtf_obs, next_portfolio, done
            )
            
            episode_reward += reward
            episode_length += 1
            episode_trades = ep_trade_count
            mtf_obs = next_mtf_obs
            portfolio = next_portfolio
            
            # Training
            if step >= self.learning_starts:
                for _ in range(self.gradient_steps):
                    metrics = self.update()
                
                if metrics:
                    metrics_history['critic_loss'].append(metrics['critic_loss'])
                    metrics_history['actor_loss'].append(metrics['actor_loss'])
                    metrics_history['alpha'].append(metrics['alpha'])
                    last_metrics = metrics
            
            # Episode end
            if done:
                episode += 1
                episode_rewards.append(episode_reward)
                episode_trade_list.append(episode_trades)
                metrics_history['episode_rewards'].append(episode_reward)
                metrics_history['episode_lengths'].append(episode_length)
                metrics_history['episode_trades'].append(episode_trades)
                
                window = episode_rewards[-10:]
                avg10 = sum(window) / len(window)
                alpha_now = last_metrics.get('alpha', self.alpha)
                closs_now = last_metrics.get('critic_loss', 0.0)
                win = "*" if episode_reward > 0 else " "
                
                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    self.save(best_model_path, save_buffer=False)
                    win = "BEST"
                
                pbar.write(
                    "{:>5}  {:>8,}  {:>+8.2f}  {:>+8.2f}  {:>+8.2f}  {:>6}  {:>7}  {:>7.4f}  {:>7.4f}  {}".format(
                        episode,
                        step + 1,
                        episode_reward,
                        avg10,
                        best_episode_reward,
                        episode_length,
                        episode_trades,
                        alpha_now,
                        closs_now,
                        win,
                    )
                )
                
                if callback:
                    callback(step, total_steps, episode_reward, episode_length, info)
                
                obs, _ = env.reset()
                mtf_obs = obs
                portfolio = obs.get('portfolio', np.zeros(5, dtype=np.float32))
                episode_reward = 0.0
                episode_length = 0
                episode_trades = 0
            
            # Checkpoint (every 10k steps by default)
            if self.total_steps - last_checkpoint_step >= checkpoint_interval:
                ckpt = os.path.join(checkpoint_dir, f'sac_mtf_checkpoint_{self.total_steps}.pt')
                self.save(ckpt)
                # Only keep latest checkpoint to save disk space
                self.save(os.path.join(checkpoint_dir, 'sac_mtf_checkpoint_latest.pt'))
                last_checkpoint_step = self.total_steps
                print(f"[SAC-MTF] Checkpoint saved at step {self.total_steps:,}")
            
            # Progress bar
            if step % 10 == 0:
                postfix = {'ep': episode, 'reward': f'{episode_reward:+.1f}'}
                if last_metrics:
                    postfix['loss'] = f'{last_metrics.get("critic_loss", 0):.2f}'
                pbar.set_postfix(postfix)
        
        # Final checkpoint
        final_path = os.path.join(checkpoint_dir, 'sac_mtf_checkpoint_final.pt')
        self.save(final_path)
        pbar.close()
        
        # Summary
        n = len(episode_rewards)
        wins = sum(1 for r in episode_rewards if r > 0)
        avg_l = (sum(metrics_history['episode_lengths']) / n) if n else 0
        avg_t = (sum(episode_trade_list) / n) if n else 0
        avg10 = (sum(episode_rewards[-10:]) / len(episode_rewards[-10:])) if n else 0
        
        print("\n" + "=" * 65)
        print("  TRAINING SUMMARY (MTF)")
        print("=" * 65)
        print("  Total episodes         : {:,}".format(n))
        print("  Total env steps        : {:,}".format(self.total_steps))
        print("  Total gradient updates : {:,}".format(self.training_step))
        print("-" * 65)
        print("  Best episode reward    : {:+.4f}".format(best_episode_reward))
        print("  Avg reward (last 10)   : {:+.4f}".format(avg10))
        print("  Win rate               : {:.1f}%  ({}/{} positive episodes)".format(
            100 * wins / n if n else 0, wins, n))
        print("  Avg episode length     : {:.1f} steps".format(avg_l))
        print("  Avg trades / episode   : {:.1f}".format(avg_t))
        print("-" * 65)
        final_alpha = last_metrics.get('alpha', self.alpha)
        final_closs = last_metrics.get('critic_loss', float('nan'))
        final_aloss = last_metrics.get('actor_loss', float('nan'))
        print("  Final alpha (entropy)  : {:.4f}".format(final_alpha))
        print("  Final critic loss      : {:.4f}".format(final_closs))
        print("  Final actor  loss      : {:.4f}".format(final_aloss))
        print("-" * 65)
        print("  Best model saved to    : {}".format(best_model_path))
        print("  Final model saved to   : {}".format(final_path))
        print("=" * 65 + "\n")
        
        metrics_history['best_episode_reward'] = best_episode_reward
        metrics_history['win_rate'] = wins / n if n else 0
        return metrics_history
    
    def save(self, path: str, save_buffer: bool = True):
        """Save MTF agent to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        checkpoint = {
            'actor_state_dict': {k: v.cpu() for k, v in self.actor.state_dict().items()},
            'critic_state_dict': {k: v.cpu() for k, v in self.critic.state_dict().items()},
            'critic_target_state_dict': {k: v.cpu() for k, v in self.critic_target.state_dict().items()},
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'total_steps': self.total_steps,
            'training_step': self.training_step,
            'mtf_config': self.mtf_config,
            'action_dim': self.action_dim,
        }
        
        torch.save(checkpoint, path)
        print(f"[SAC-MTF] Model saved to: {path}")
    
    def load(self, path: str):
        """Load MTF agent from file."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        log_alpha_val = checkpoint['log_alpha'].item()
        self.log_alpha = torch.tensor(
            [log_alpha_val], dtype=torch.float32,
            requires_grad=True, device=self.device
        )
        self.alpha = float(torch.exp(self.log_alpha).item())
        
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=self.mtf_config.get('alpha_lr', 3e-4)
        )
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.training_step = checkpoint.get('training_step', 0)
        
        print(f"[SAC-MTF] Model loaded from: {path}")
        print(f"      device={self.device}  steps={self.total_steps}")
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = 'auto'):
        """Load MTF agent from saved checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        mtf_config = checkpoint['mtf_config']
        action_dim = checkpoint['action_dim']
        
        agent = cls(mtf_config, action_dim, device)
        agent.load(path)
        
        return agent


# ============================================================
# Legacy SAC Agent (for backward compatibility)
# ============================================================

class SACAgent:
    """
    Legacy single-timeframe SAC Agent.
    Kept for backward compatibility with existing trained models.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: dict,
        device: str = 'auto',
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        if device == 'auto':
            self.device = self._get_device()
        else:
            self.device = device
            
        print(f"[SAC] Using device: {self.device}")
        
        self.actor, self.critic = build_networks(state_dim, action_dim, config)
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        _, self.critic_target = build_networks(state_dim, action_dim, config)
        self.critic_target.to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        for param in self.critic_target.parameters():
            param.requires_grad = False
            
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.get('actor_lr', config.get('learning_rate', 3e-4))
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.get('critic_lr', config.get('learning_rate', 3e-4))
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=config.get('alpha_lr', 3e-4)
        )
        
        self.buffer = create_buffer(config, self.device,
                                    state_dim=state_dim, action_dim=action_dim)
        
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 256)
        self.learning_starts = config.get('learning_starts', 1000)
        self.use_per = config.get('buffer_type') == 'prioritized'
        self.gradient_steps = config.get('gradient_steps', 1)
        
        self.total_steps = 0
        self.training_step = 0
        
    def _get_device(self) -> str:
        """Auto-detect best available device (CUDA > CPU)."""
        if torch.cuda.is_available():
            print(f"[SAC] CUDA available: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        print("[SAC] Using CPU (no GPU detected)")
        return 'cpu'
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor(state_tensor, deterministic=False)
        return action.cpu().numpy()[0]
    
    def select_action_deterministic(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor(state_tensor, deterministic=True)
        return action.cpu().numpy()[0]
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray, done: bool):
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        
    def update(self) -> Optional[Dict[str, float]]:
        if not self.buffer.is_ready(self.batch_size):
            return None
            
        if self.use_per:
            batch = self.buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, indices, weights = batch
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            indices, weights = None, None
            
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states, deterministic=False)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target_min = torch.min(q1_target, q2_target)
            q_target = rewards + (1 - dones) * self.gamma * (
                q_target_min - self.alpha * next_log_probs
            )
            
        q1_current, q2_current = self.critic(states, actions)
        q_target = torch.clamp(q_target, -100.0, 100.0)

        if weights is not None:
            critic1_loss = (weights * F.smooth_l1_loss(q1_current, q_target, reduction='none')).mean()
            critic2_loss = (weights * F.smooth_l1_loss(q2_current, q_target, reduction='none')).mean()
            td_errors = ((q1_current - q_target).abs() + (q2_current - q_target).abs()) / 2
        else:
            critic1_loss = F.smooth_l1_loss(q1_current, q_target)
            critic2_loss = F.smooth_l1_loss(q2_current, q_target)

        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        if self.use_per and indices is not None:
            td_errors_np = td_errors.detach().cpu().numpy().flatten()
            self.buffer.update_priorities(indices, td_errors_np)
            
        new_actions, log_probs = self.actor(states, deterministic=False)
        log_probs = torch.clamp(log_probs, -10, 10)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new_min = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new_min).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 0.5)
        self.alpha_optimizer.step()
        
        self.log_alpha.data = torch.clamp(self.log_alpha.data, -5, 2)
        self.alpha = self.log_alpha.exp().item()
        
        self._soft_update(self.critic, self.critic_target, self.tau)
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
            'q1_mean': q1_current.mean().item(),
            'q2_mean': q2_current.mean().item(),
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )
            
    def learn(self, total_steps: int, env, callback=None,
              checkpoint_dir: str = 'models', checkpoint_interval: int = 50_000,
              best_model_path: Optional[str] = None) -> Dict[str, Any]:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if best_model_path is None:
            best_model_path = os.path.join(checkpoint_dir, 'sac_best.pt')

        state, _ = env.reset()
        episode = 0
        episode_reward = 0.0
        episode_length = 0
        episode_trades = 0
        episode_rewards = []
        episode_trade_list = []
        best_episode_reward = -float('inf')
        metrics_history = {
            'episode_rewards': [], 'episode_lengths': [], 'episode_trades': [],
            'critic_loss': [], 'actor_loss': [], 'alpha': [],
        }
        pbar = tqdm(range(total_steps), desc="Training", unit="step",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        last_metrics = {}
        last_checkpoint_step = 0

        print("\n{:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}  {:>7}  {:>7}  {}".format(
            "Ep", "Step", "Reward", "Avg10", "Best", "Len", "Trades", "Alpha", "CriticL"))
        print("-" * 75)

        for step in pbar:
            if step < self.learning_starts:
                action = env.action_space.sample()
            else:
                action = self.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_trade_count = info.get('trade_count', 0)

            self.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            episode_length += 1
            episode_trades = ep_trade_count
            state = next_state

            if step >= self.learning_starts:
                for _ in range(self.gradient_steps):
                    metrics = self.update()
                if metrics:
                    metrics_history['critic_loss'].append(metrics['critic_loss'])
                    metrics_history['actor_loss'].append(metrics['actor_loss'])
                    metrics_history['alpha'].append(metrics['alpha'])
                    last_metrics = metrics

            if done:
                episode += 1
                episode_rewards.append(episode_reward)
                episode_trade_list.append(episode_trades)
                metrics_history['episode_rewards'].append(episode_reward)
                metrics_history['episode_lengths'].append(episode_length)
                metrics_history['episode_trades'].append(episode_trades)

                window = episode_rewards[-10:]
                avg10 = sum(window) / len(window)
                alpha_now = last_metrics.get('alpha', self.alpha)
                closs_now = last_metrics.get('critic_loss', 0.0)
                win = "*" if episode_reward > 0 else " "

                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    self.save(best_model_path, save_buffer=False)
                    win = "BEST"

                pbar.write("{:>5}  {:>8,}  {:>+8.2f}  {:>+8.2f}  {:>+8.2f}  {:>6}  {:>7}  {:>7.4f}  {:>7.4f}  {}".format(
                    episode, step + 1, episode_reward, avg10, best_episode_reward,
                    episode_length, episode_trades, alpha_now, closs_now, win))

                if callback:
                    callback(step, total_steps, episode_reward, episode_length, info)

                state, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_trades = 0

            if self.total_steps - last_checkpoint_step >= checkpoint_interval:
                ckpt = os.path.join(checkpoint_dir, f'sac_checkpoint_{self.total_steps}.pt')
                self.save(ckpt)
                self.save(os.path.join(checkpoint_dir, 'sac_checkpoint_latest.pt'))
                last_checkpoint_step = self.total_steps

            if step % 10 == 0:
                postfix = {'ep': episode, 'reward': f'{episode_reward:+.1f}'}
                if last_metrics:
                    postfix['loss'] = f'{last_metrics.get("critic_loss", 0):.2f}'
                pbar.set_postfix(postfix)

        final_path = os.path.join(checkpoint_dir, 'sac_checkpoint_final.pt')
        self.save(final_path)
        pbar.close()

        n = len(episode_rewards)
        wins = sum(1 for r in episode_rewards if r > 0)
        avg_l = (sum(metrics_history['episode_lengths']) / n) if n else 0
        avg_t = (sum(episode_trade_list) / n) if n else 0
        avg10 = (sum(episode_rewards[-10:]) / len(episode_rewards[-10:])) if n else 0

        print("\n" + "=" * 65)
        print("  TRAINING SUMMARY")
        print("=" * 65)
        print("  Total episodes         : {:,}".format(n))
        print("  Total env steps        : {:,}".format(self.total_steps))
        print("  Total gradient updates : {:,}".format(self.training_step))
        print("-" * 65)
        print("  Best episode reward    : {:+.4f}".format(best_episode_reward))
        print("  Avg reward (last 10)   : {:+.4f}".format(avg10))
        print("  Win rate               : {:.1f}%  ({}/{} positive episodes)".format(
            100 * wins / n if n else 0, wins, n))
        print("  Avg episode length     : {:.1f} steps".format(avg_l))
        print("  Avg trades / episode   : {:.1f}".format(avg_t))
        print("-" * 65)
        final_alpha = last_metrics.get('alpha', self.alpha)
        final_closs = last_metrics.get('critic_loss', float('nan'))
        final_aloss = last_metrics.get('actor_loss', float('nan'))
        print("  Final alpha (entropy)  : {:.4f}".format(final_alpha))
        print("  Final critic loss      : {:.4f}".format(final_closs))
        print("  Final actor  loss      : {:.4f}".format(final_aloss))
        print("-" * 65)
        print("  Best model saved to    : {}".format(best_model_path))
        print("  Final model saved to   : {}".format(final_path))
        print("=" * 65 + "\n")

        metrics_history['best_episode_reward'] = best_episode_reward
        metrics_history['win_rate'] = wins / n if n else 0
        return metrics_history
    
    def save(self, path: str, save_buffer: bool = True):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        checkpoint = {
            'actor_state_dict': {k: v.cpu() for k, v in self.actor.state_dict().items()},
            'critic_state_dict': {k: v.cpu() for k, v in self.critic.state_dict().items()},
            'critic_target_state_dict': {k: v.cpu() for k, v in self.critic_target.state_dict().items()},
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'total_steps': self.total_steps,
            'training_step': self.training_step,
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }
        
        torch.save(checkpoint, path)
        print(f"[SAC] Model saved to: {path}")
        
        if save_buffer and len(self.buffer) > 0:
            buffer_path = path.replace('.pt', '_buffer.pkl').replace('.zip', '_buffer.pkl')
            self.buffer.save(buffer_path)
        
    def load(self, path: str, load_buffer: bool = True):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        log_alpha_val = checkpoint['log_alpha'].item()
        self.log_alpha = torch.tensor(
            [log_alpha_val], dtype=torch.float32,
            requires_grad=True, device=self.device
        )
        self.alpha = float(torch.exp(self.log_alpha).item())
        
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=self.config.get('alpha_lr', 3e-4)
        )
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.training_step = checkpoint.get('training_step', 0)
        
        print(f"[SAC] Model loaded from: {path}")
        print(f"      device={self.device}  steps={self.total_steps}")
        
        if load_buffer:
            buffer_path = path.replace('.pt', '_buffer.pkl').replace('.zip', '_buffer.pkl')
            if os.path.exists(buffer_path):
                self.buffer.load(buffer_path)
        
    @classmethod
    def from_pretrained(cls, path: str, device: str = 'auto'):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        state_dim = checkpoint['state_dim']
        action_dim = checkpoint['action_dim']
        
        agent = cls(state_dim, action_dim, config, device)
        agent.load(path)
        
        return agent