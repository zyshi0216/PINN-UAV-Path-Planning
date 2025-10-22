# pinn_uav_2d.py
# Minimal working 2D-PINN for UAV path planning in a dynamic wind field
# Dependencies: torch, numpy, matplotlib

import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ========== Config ==========
class Cfg:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # Time normalization: t in [0,1]
    T = 1.0

    # Start/Goal (meters)
    x0, y0 = 0.0, 0.0
    vx0, vy0 = 0.0, 0.0
    xT, yT = 10.0, 6.0
    vxT, vyT = 0.0, 0.0

    # Drag
    c_d = 0.2

    # Wind field parameters (simple, smooth, time-varying)
    W_Ax = 1.2
    W_Ay = 0.8

    # Obstacle: circle (cx, cy, r)
    obstacles = [(5,3,1.2), (7,2,1.0), (3,2,0.8)]


    # Loss weights
    lam_phys = 1.0
    lam_bc = 10.0
    lam_obj = 0.5
    # Objective terms
    alpha_energy = 1.0    # control energy
    beta_smooth = 0.05    # control temporal smoothness
    gamma_risk = 15.0      # obstacle risk weight

    # Sampling / training
    N_col = 2048           # collocation points per epoch
    N_bc = 256             # boundary samples per epoch (t=0,t=1 mixed)
    epochs = 6000
    lr = 1e-3
    print_every = 200

    # Export
    export_n = 400
    out_csv = "traj.csv"
    out_fig = "traj.jpg"

cfg = Cfg()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# ========== Wind Field ==========
def wind_field(x, y, t):
    """
    Smooth, time-varying wind (Wx, Wy).
    x,y,t are tensors of same shape.
    """
    # Example: coupled sine fields with time modulation
    Wx = cfg.W_Ax * torch.sin(math.pi * y / 6.0) * torch.cos(2 * math.pi * t)
    Wy = cfg.W_Ay * torch.sin(math.pi * x / 10.0) * torch.sin(2 * math.pi * t)
    return Wx, Wy

# ========== Obstacles ==========
def obstacle_risk(x, y):
    """
    基于“1/距离^2”的势垒函数；离障碍边界越近惩罚越大。
    对每个圆障碍：d = 距离圆心 - 半径。d>0 在外部；d≈0 附近梯度最大。
    """
    eps = 1e-4
    total = torch.zeros_like(x)
    for (cx, cy, r) in cfg.obstacles:
        d = torch.sqrt((x - cx)**2 + (y - cy)**2 + eps) - r
        # 外部：1/d^2；内部：对 d<0 额外加一个软势垒（softplus），防止穿越
        outside_term = 1.0 / (d**2 + eps)
        inside_barrier = torch.nn.functional.softplus(-10.0 * d)  # 进入内部惩罚迅速抬升
        total = total + outside_term + inside_barrier
    return total


# ========== PINN Model ==========
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def mlp(in_dim, hidden, out_dim, act=Sine()):
    layers = []
    width = hidden
    layers += [nn.Linear(in_dim, width), act]
    layers += [nn.Linear(width, width), act]
    layers += [nn.Linear(width, width), act]
    layers += [nn.Linear(width, width), act]
    layers += [nn.Linear(width, out_dim)]
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net

class PINN_UAV(nn.Module):
    """
    Input: t in [0,1], shape [N,1]
    Output: concat [x,y,vx,vy, ux,uy], shape [N,6]
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.net = mlp(1, hidden, 6, act=Sine())

    def forward(self, t):
        return self.net(t)

# ========== Physics residual ==========
def physics_residual(model, t):
    """
    物理残差：
      x' = vx
      y' = vy
      vx' = ux - c_d * vx + Wx
      vy' = uy - c_d * vy + Wy
    """
    t = t.requires_grad_(True)
    out = model(t)
    x, y, vx, vy, ux, uy = out[:,0], out[:,1], out[:,2], out[:,3], out[:,4], out[:,5]

    # 单独对每个量求导，避免多输出 grad 返回值合并的问题
    dx_dt  = torch.autograd.grad(x,  t, grad_outputs=torch.ones_like(x),  create_graph=True, retain_graph=True)[0].squeeze(-1)
    dy_dt  = torch.autograd.grad(y,  t, grad_outputs=torch.ones_like(y),  create_graph=True, retain_graph=True)[0].squeeze(-1)
    dvx_dt = torch.autograd.grad(vx, t, grad_outputs=torch.ones_like(vx), create_graph=True, retain_graph=True)[0].squeeze(-1)
    dvy_dt = torch.autograd.grad(vy, t, grad_outputs=torch.ones_like(vy), create_graph=True, retain_graph=True)[0].squeeze(-1)

    Wx, Wy = wind_field(x, y, t.squeeze(-1))

    r1 = dx_dt - vx
    r2 = dy_dt - vy
    r3 = dvx_dt - (ux - cfg.c_d * vx + Wx)
    r4 = dvy_dt - (uy - cfg.c_d * vy + Wy)

    return (r1**2 + r2**2 + r3**2 + r4**2).mean()

# ========== Boundary conditions ==========
def boundary_loss(model, N_bc):
    """
    Sample t=0 and t=1 points; enforce start/end position & (near-)zero velocity.
    """
    # Build batch: half for t=0, half for t=1
    n_half = N_bc // 2
    t0 = torch.zeros((n_half,1), device=cfg.device)
    t1 = torch.ones((N_bc - n_half,1), device=cfg.device)
    tt = torch.cat([t0, t1], dim=0)

    out = model(tt)
    x, y, vx, vy, ux, uy = out[:,0], out[:,1], out[:,2], out[:,3], out[:,4], out[:,5]

    loss0 = (x[:n_half]-cfg.x0).pow(2).mean() + (y[:n_half]-cfg.y0).pow(2).mean() \
          + (vx[:n_half]-cfg.vx0).pow(2).mean() + (vy[:n_half]-cfg.vy0).pow(2).mean()

    loss1 = (x[n_half:]-cfg.xT).pow(2).mean() + (y[n_half:]-cfg.yT).pow(2).mean() \
          + (vx[n_half:]-cfg.vxT).pow(2).mean() + (vy[n_half:]-cfg.vyT).pow(2).mean()

    return loss0 + loss1

# ========== Objective (energy + smoothness + risk) ==========
def objective_loss(model, t):
    """
    目标项：能耗 + 控制平滑 + 障碍风险
      alpha * ||u||^2 + beta * ||du/dt||^2 + gamma * Risk(x,y)
    """
    t = t.requires_grad_(True)
    out = model(t)
    x, y, vx, vy, ux, uy = out[:,0], out[:,1], out[:,2], out[:,3], out[:,4], out[:,5]

    energy = (ux**2 + uy**2).mean()

    # 单独对 ux, uy 求导
    dux_dt = torch.autograd.grad(ux, t, grad_outputs=torch.ones_like(ux), create_graph=True, retain_graph=True)[0].squeeze(-1)
    duy_dt = torch.autograd.grad(uy, t, grad_outputs=torch.ones_like(uy), create_graph=True, retain_graph=True)[0].squeeze(-1)
    smooth = (dux_dt**2 + duy_dt**2).mean()

    raw_risk = obstacle_risk(x, y)
    # 防止极端情况下的爆炸，做一个轻微截断
    risk = torch.clamp(raw_risk, max=1e3).mean()


    L = cfg.alpha_energy * energy + cfg.beta_smooth * smooth + cfg.gamma_risk * risk
    return L, energy.item(), smooth.item(), risk.item()

# ========== Training ==========
def train():
    model = PINN_UAV(hidden=128).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.epochs+1):
        # Collocation points
        t_col = torch.rand((cfg.N_col,1), device=cfg.device)  # U(0,1)
        L_phys = physics_residual(model, t_col)

        # Boundary
        L_bc = boundary_loss(model, cfg.N_bc)

        # Objective (use same t or resample)
        t_obj = torch.rand((cfg.N_col,1), device=cfg.device)
        L_obj, e_val, s_val, r_val = objective_loss(model, t_obj)

        # Composite
        loss = cfg.lam_phys * L_phys + cfg.lam_bc * L_bc + cfg.lam_obj * L_obj

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % cfg.print_every == 0 or ep == 1:
            print(f"[{ep:5d}] "
                  f"Loss={loss.item():.4e}  "
                  f"Phys={L_phys.item():.3e}  BC={L_bc.item():.3e}  Obj={L_obj.item():.3e}  "
                  f"(E={e_val:.3e}, S={s_val:.3e}, R={r_val:.3e})")

        # Simple curriculum: gradually increase physics weight
                # ----- curriculum: 逐步提高 physics 与 objective 的权重 -----
        if ep % 2000 == 0 and ep < cfg.epochs:
            cfg.lam_phys *= 1.3                  # 物理一致性越来越硬
            cfg.lam_obj  = min(cfg.lam_obj*1.3, 0.6)  # 目标项也逐步加强，上限 0.6
            # 同时可小幅上调 gamma_risk（可选）
            cfg.gamma_risk = min(cfg.gamma_risk * 1.1, 30.0)


    return model

# ========== Export / Visualization ==========
def export(model):
    model.eval()
    with torch.no_grad():
        t = torch.linspace(0., 1., cfg.export_n, device=cfg.device).view(-1,1)
        out = model(t)
        x = out[:,0].cpu().numpy()
        y = out[:,1].cpu().numpy()
        vx = out[:,2].cpu().numpy()
        vy = out[:,3].cpu().numpy()
        ux = out[:,4].cpu().numpy()
        uy = out[:,5].cpu().numpy()
        tt = t.squeeze(-1).cpu().numpy()

    # Save CSV
    data = np.stack([tt, x, y, vx, vy, ux, uy], axis=1)
    np.savetxt(cfg.out_csv, data, delimiter=",", header="t,x,y,vx,vy,ux,uy", comments="")
    print(f"[OK] Saved CSV: {cfg.out_csv}")

    # Plot
    fig, ax = plt.subplots(figsize=(7,4.5), dpi=140)
    ax.plot(x, y, lw=2)
    ax.scatter([cfg.x0, cfg.xT], [cfg.y0, cfg.yT], c=["green","red"], s=60, zorder=5, label="Start/Goal")
    ax.text(cfg.x0, cfg.y0, "  Start", va="center")
    ax.text(cfg.xT, cfg.yT, "  Goal", va="center")

    # Obstacles
    for (cx, cy, r) in cfg.obstacles:
        circ = plt.Circle((cx,cy), r, color="gray", alpha=0.25)
        ax.add_patch(circ)
        ax.plot(cx, cy, "kx", ms=6)

    # Wind quiver (sparse grid)
    Xg, Yg = np.meshgrid(np.linspace(min(x.min(),0)-1, max(x.max(),cfg.xT)+1, 20),
                         np.linspace(min(y.min(),0)-1, max(y.max(),cfg.yT)+1, 14))
    Tg = np.full_like(Xg, 0.5)  # show mid-time slice
    Xgt = torch.from_numpy(Xg).float().to(cfg.device)
    Ygt = torch.from_numpy(Yg).float().to(cfg.device)
    Tgt = torch.from_numpy(Tg).float().to(cfg.device)
    Wx, Wy = wind_field(Xgt, Ygt, Tgt)
    Wx = Wx.cpu().numpy(); Wy = Wy.cpu().numpy()
    ax.quiver(Xg, Yg, Wx, Wy, alpha=0.35, pivot="mid", angles="xy", scale_units="xy", scale=5.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("PINN UAV Trajectory (2D, dynamic wind)")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(cfg.out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Figure: {cfg.out_fig}")

def main():
    print(f"Device: {cfg.device}")
    model = train()
    export(model)

if __name__ == "__main__":
    main()
