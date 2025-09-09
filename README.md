# AdaptiveScheduler

here’s a **README.md** you can drop straight into a GitHub repo for your `AdaptiveScheduler`. it blends the theory you’ve worked out with clear install / usage docs, examples, and notes.

---

````markdown
# 🔀 AdaptiveScheduler

An **adaptive, signal-driven learning rate & momentum scheduler** for PyTorch.  
It works with **any optimizer** (AdamW, SGD, Adafactor, etc.) and adapts hyperparameters based on:

- **Loss** dynamics (decay, trend, variance)
- **Gradient norm** feedback
- **Reward** signals (for RL fine-tuning, e.g. Hugging Face TRL)
- **Cyclical resets**

👉 Unlike built-in schedulers, `AdaptiveScheduler` is **agnostic**: it doesn’t assume time-only decay.  
You explicitly call `observe(loss=..., R=..., model=...)` after each `optimizer.step()`.

---

## ✨ Features

- Works with **any optimizer**
- Multiple rules for LR:
  - `loss_decay`, `trend`, `variance`, `grad_norm`, `cyclical`
- Momentum rules (for SGD-like):
  - `inverse`, `exp`, `relative`, `normalized`
- Reward rules:
  - `scaling`, `trend`, `variance`
- **Clamps** keep LR/momentum in safe ranges
- Optional **EMA smoothing** for noisy signals
- Integrates with [Hugging Face TRL](https://github.com/huggingface/trl) for RL training
- Compatible with logging tools (`get_lr()` mirrors PyTorch `_LRScheduler`)

---

## 📦 Installation

Clone the repo and install locally:

```bash
git clone https://github.com/yourname/adaptive-scheduler.git
cd adaptive-scheduler
pip install -e .
````

Requires:

* `torch >= 1.13`
* `numpy`

---

## 🚀 Quickstart

```python
import torch
from torch.optim import AdamW
from adaptive_scheduler import AdaptiveScheduler

model = ...
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# build adaptive scheduler
sched = AdaptiveScheduler(
    optimizer,
    lr_rule="trend",          # or "loss_decay", "variance", "grad_norm", "cyclical"
    momentum_rule="none",     # e.g. "relative" if using SGD
    reward_rule="trend",      # "scaling" | "trend" | "variance"
    ema_reward_beta=0.9,      # smooth noisy rewards
    min_lr=1e-6, max_lr=5e-4,
)

for batch in dataloader:
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()

    # handshake with scheduler
    reward = compute_reward(batch)  # optional
    sched.observe(loss=None, R=reward, model=None)

    optimizer.zero_grad()
```

---

## 🤝 Integration with TRL (Hugging Face RL)

```python
from trl import ReinforceTrainer, ReinforceConfig
from torch.optim import AdamW
from adaptive_scheduler import AdaptiveScheduler

optimizer = AdamW(model.parameters(), lr=5e-5)
sched = AdaptiveScheduler(optimizer, lr_rule="none", reward_rule="trend")

# wrap optimizer.step() once so scheduler auto-updates
import types
_orig_step = optimizer.step
def _wrapped_step(*a, **kw):
    out = _orig_step(*a, **kw)
    sched.observe(loss=None, R=last_reward(), model=None)
    return out
optimizer.step = types.MethodType(_wrapped_step, optimizer)

trainer = ReinforceTrainer(
    model=model,
    args=ReinforceConfig(),
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_fn=my_reward_fn,   # your reward function should record to last_reward()
)
trainer.optimizer = optimizer
trainer.lr_scheduler = sched
trainer.train()
```

---

## 📐 Theory

Signals per step:

* Loss $\ell_t$, Reward $R_t$, Grad norm $g_t$
* LR update examples:

  * LossDecay: $\eta_{t+1} = \eta_t / (1 + k\ell_t)$
  * Trend: increase if loss improves, decrease otherwise
  * RewardTrend: LR up if reward improves, down otherwise
* Momentum update examples:

  * Inverse: $\mu_{t+1} = 1 - 1/(1+\ell_t)$
  * Exp: $\mu_{t+1} = 1 - e^{-a\ell_t}$

Safety:

* Always clamp to `[min_lr, max_lr]`, `[min_momentum, max_momentum]`.
* If all rules = `"none"`, scheduler is inert (identity).

---

## ✅ Unit Test Checklist

* [ ] LR never leaves bounds
* [ ] Momentum only updates for optimizers with momentum
* [ ] RewardScaling maps `(center±span)` to `(≈0, ≈2*base_lr)`
* [ ] EMA smoothing behaves as expected
* [ ] `state_dict()` round-trips

---

## 📊 Benchmarks

Recommended first tasks (with TRL):

* **Sentiment-controlled generation** (IMDB) — reward = classifier score
* **Summarization length control** — reward = length penalty

Compare vs:

* Constant LR
* CosineAnnealing
* OneCycleLR
* ReduceLROnPlateau

Metrics:

* Avg reward ↑
* Reward variance ↓
* KL divergence stability
* Sample efficiency (reward vs steps)

---

## ⚖️ License

MIT

---

## 🙌 Contributing

Issues and PRs welcome! Please add tests for new rules or hyperparameters.

```

---

do you want me to also draft a **minimal `examples/` script** (e.g. `examples/train_with_rewards.py`) that users can run end-to-end with TRL + AdaptiveScheduler?
```
