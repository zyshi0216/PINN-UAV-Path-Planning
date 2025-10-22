# make_mosaic.py
# 生成单图 + 拼接马赛克（3x4），每格有大号简写标签，导出 PNG/PDF
# 依赖：numpy, pandas, matplotlib, pillow

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

TRAJ_CSV = "traj.csv"
OUTDIR = "paper_figs"              # 单图输出目录
MOS_PNG = "figure_mosaic.png"      # 拼图输出
MOS_PDF = "figure_mosaic.pdf"

# —— 障碍/风场参数（按你训练时的设定修改）——
OBSTACLES = [(3.0, 2.0, 1.0), (5.0, 3.0, 1.5), (7.5, 4.0, 1.2)]
W_AX, W_AY = 1.2, 0.8

# ===== 公共工具 =====
def finite_diff(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    dy = np.empty_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2] + 1e-12)
    dy[0]    = (y[1]  - y[0])   / (x[1]  - x[0]   + 1e-12)
    dy[-1]   = (y[-1] - y[-2])  / (x[-1] - x[-2]  + 1e-12)
    return dy

def curvature(x, y, t):
    x1, y1 = finite_diff(x, t), finite_diff(y, t)
    x2, y2 = finite_diff(x1, t), finite_diff(y1, t)
    num = np.abs(x1*y2 - y1*x2)
    den = (x1**2 + y1**2)**1.5 + 1e-12
    return num/den

def safety_margin(x, y, obstacles):
    mats = []
    for (cx,cy,r) in obstacles:
        d = np.sqrt((x-cx)**2 + (y-cy)**2) - r
        mats.append(d)
    return np.max(np.vstack(mats), axis=0)

def wind_field(x, y, t):
    Wx = W_AX * np.sin(np.pi * y / 6.0) * np.cos(2*np.pi*t)
    Wy = W_AY * np.sin(np.pi * x / 10.0) * np.sin(2*np.pi*t)
    return Wx, Wy

def cumtrapz(y, x):
    acc = np.zeros_like(y)
    for i in range(1, len(y)):
        acc[i] = acc[i-1] + 0.5*(y[i]+y[i-1])*(x[i]-x[i-1])
    return acc

def save_line(x, y, xlabel, ylabel, title, out, legend=None):
    plt.figure(figsize=(7,4.5), dpi=160)
    plt.plot(x, y, linewidth=2, label=legend if legend else None)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    if legend: plt.legend()
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()

# ===== 读取轨迹 =====
if not os.path.exists(TRAJ_CSV):
    raise FileNotFoundError(f"{TRAJ_CSV} not found.")
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(TRAJ_CSV)
for c in ["t","x","y","vx","vy","ux","uy"]:
    if c not in df.columns: raise ValueError(f"Missing column: {c}")
t,x,y,vx,vy,ux,uy = (df[c].to_numpy() for c in ["t","x","y","vx","vy","ux","uy"])

# ===== 派生指标 =====
speed   = np.sqrt(vx**2 + vy**2)
ax, ay  = finite_diff(vx,t), finite_diff(vy,t)
accel   = np.sqrt(ax**2 + ay**2)
u_mag   = np.sqrt(ux**2 + uy**2)
dux, duy= finite_diff(ux,t), finite_diff(uy,t)
udot    = np.sqrt(dux**2 + duy**2)
curv    = curvature(x,y,t)
margin  = safety_margin(x,y,OBSTACLES)
Wx, Wy  = wind_field(x,y,t)
vdotW   = vx*Wx + vy*Wy
E_cum   = cumtrapz(u_mag**2, t)
L_cum   = cumtrapz(speed, t)

# ===== 单图生成（论文风格：一图一轴，无子图）=====
# 1 Trajectory（时间线宽渐变 + 障碍）
lw = np.linspace(1.5, 3.2, len(x))
plt.figure(figsize=(7,4.5), dpi=160)
for i in range(1,len(x)):
    plt.plot(x[i-1:i+1], y[i-1:i+1], linewidth=lw[i])
for (cx,cy,r) in OBSTACLES:
    circ = plt.Circle((cx,cy), r, alpha=0.15)
    plt.gca().add_patch(circ); plt.plot(cx,cy,marker="x",linestyle="None")
plt.xlabel("x (m)"); plt.ylabel("y (m)"); plt.title("UAV Trajectory")
plt.grid(True, linestyle="--", alpha=0.3); plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"fig_traj.png"), bbox_inches="tight"); plt.close()

# 其余时序图
save_line(t, speed, "t", "|v| (m/s)",       "Speed vs Time",              os.path.join(OUTDIR,"fig_speed.png"))
save_line(t, accel, "t", "|a| (m/s^2)",     "Acceleration vs Time",       os.path.join(OUTDIR,"fig_accel.png"))
plt.figure(figsize=(7,4.5), dpi=160)
plt.plot(t, ux, linewidth=1.8, label="u_x")
plt.plot(t, uy, linewidth=1.8, label="u_y")
plt.plot(t, u_mag, linewidth=1.8, linestyle="--", label="|u|")
plt.xlabel("t"); plt.ylabel("u"); plt.title("Control Inputs vs Time")
plt.grid(True, linestyle="--", alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"fig_ctrl.png"), bbox_inches="tight"); plt.close()
save_line(t, udot, "t", "|du/dt|",          "Control Rate vs Time",       os.path.join(OUTDIR,"fig_dudt.png"))
save_line(t, E_cum, "t", "∫||u||² dt",      "Cumulative Control Energy",  os.path.join(OUTDIR,"fig_Ecum.png"))
save_line(t, L_cum, "t", "∫||v|| dt (m)",   "Cumulative Path Length",     os.path.join(OUTDIR,"fig_Lcum.png"))
save_line(t, curv,  "t", "κ",               "Curvature vs Time",          os.path.join(OUTDIR,"fig_curv.png"))
save_line(t, margin,"t", "Margin (m)",      "Safety Margin vs Time",      os.path.join(OUTDIR,"fig_margin.png"))
plt.figure(figsize=(7,4.5), dpi=160)
plt.plot(vx, vy, linewidth=1.8)
plt.xlabel("v_x"); plt.ylabel("v_y"); plt.title("Phase Portrait: v")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"fig_phase_v.png"), bbox_inches="tight"); plt.close()
plt.figure(figsize=(7,4.5), dpi=160)
plt.plot(ux, uy, linewidth=1.8)
plt.xlabel("u_x"); plt.ylabel("u_y"); plt.title("Control Phase: u")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"fig_phase_u.png"), bbox_inches="tight"); plt.close()
heading = np.degrees(np.arctan2(vy, vx))
save_line(t, heading, "t", "ψ (deg)",       "Heading vs Time",            os.path.join(OUTDIR,"fig_heading.png"))
save_line(t, vdotW,  "t", "v·W",            "Velocity–Wind Alignment",    os.path.join(OUTDIR,"fig_vdotW.png"))

# ===== 拼图（PIL，无子图）=====
# 选择 12 张核心图（3x4）
tiles = [
    ("fig_traj.png",   "Traj"),
    ("fig_speed.png",  "Speed"),
    ("fig_ctrl.png",   "Ctrl"),
    ("fig_dudt.png",   "dU/dt"),
    ("fig_accel.png",  "Accel"),
    ("fig_curv.png",   "Curv"),
    ("fig_margin.png", "Margin"),
    ("fig_heading.png","Head"),
    ("fig_Ecum.png",   "E_cum"),
    ("fig_Lcum.png",   "L_cum"),
    ("fig_phase_v.png","Phase v"),
    ("fig_phase_u.png","Phase u"),
]
paths = [os.path.join(OUTDIR, f) for f,_ in tiles]
for p in paths:
    if not os.path.exists(p): raise FileNotFoundError(f"Missing panel: {p}")

# 读取并统一缩放
imgs = [Image.open(p).convert("RGB") for p in paths]
# 统一成宽 W×高 H（保证文字清晰；数值可按需要调大）
W, H = 900, 560
imgs = [im.resize((W,H), Image.LANCZOS) for im in imgs]

# 画布：3列×4行，留出外边距与格间距
COLS, ROWS = 3, 4
pad = 36         # 外边距
gap = 24         # 图间距
MOS_W = pad*2 + COLS*W + (COLS-1)*gap
MOS_H = pad*2 + ROWS*H + (ROWS-1)*gap
mos = Image.new("RGB", (MOS_W, MOS_H), (255,255,255))
draw = ImageDraw.Draw(mos)

# 字体（找不到 TTF 则退回默认）
try:
    # 若你有更好的字体，替换路径，例如 Windows 的 "arial.ttf"
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default()

# 粘贴 + 左上角打简写标签（半透明底条）
for idx, (im, (_, label)) in enumerate(zip(imgs, tiles)):
    r = idx // COLS; c = idx % COLS
    x0 = pad + c*(W+gap)
    y0 = pad + r*(H+gap)
    mos.paste(im, (x0, y0))
    # 半透明条
    bar_h = 56
    bar = Image.new("RGBA", (W, bar_h), (255,255,255,200))
    mos.paste(bar, (x0, y0), bar)
    # 文本
    draw.text((x0+16, y0+10), label, fill=(0,0,0), font=font)

# 导出
mos.save(MOS_PNG)
try:
    mos.save(MOS_PDF)
except Exception as e:
    print(f"[warn] PDF export failed: {e}")
print(f"[OK] Saved mosaic: {MOS_PNG} (+PDF if available)")
