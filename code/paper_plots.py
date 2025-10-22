# paper_plots.py  —— PINN 轨迹论文图批量生成（无 seaborn、单图单轴、科研风格）
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

TRAJ_CSV = "traj.csv"
OBSTACLES = [(3.0, 2.0, 1.0), (5.0, 3.0, 1.5), (7.5, 4.0, 1.2)]
W_AX, W_AY = 1.2, 0.8  # wind params, keep consistent with training

def wind_field(x, y, t):
    Wx = W_AX * np.sin(np.pi * y / 6.0) * np.cos(2*np.pi*t)
    Wy = W_AY * np.sin(np.pi * x / 10.0) * np.sin(2*np.pi*t)
    return Wx, Wy

def finite_diff(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    dy = np.empty_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2] + 1e-12)
    dy[0]    = (y[1]  - y[0])   / (x[1]  - x[0]   + 1e-12)
    dy[-1]   = (y[-1] - y[-2])  / (x[-1] - x[-2]  + 1e-12)
    return dy

def curvature(x, y, t):
    x1 = finite_diff(x, t); y1 = finite_diff(y, t)
    x2 = finite_diff(x1, t); y2 = finite_diff(y1, t)
    num = np.abs(x1*y2 - y1*x2)
    den = (x1**2 + y1**2)**1.5 + 1e-12
    return num/den

def safety_margin(x, y, obstacles):
    margins = []
    for (cx,cy,r) in obstacles:
        d = np.sqrt((x-cx)**2 + (y-cy)**2) - r
        margins.append(d)
    return np.max(np.vstack(margins), axis=0)

def cumtrapz(y, x):
    acc = np.zeros_like(y)
    for i in range(1, len(y)):
        acc[i] = acc[i-1] + 0.5*(y[i]+y[i-1])*(x[i]-x[i-1])
    return acc

def save_line(x, y, xlabel, ylabel, title, out, legend=None):
    plt.figure(figsize=(7,4.5), dpi=140)
    plt.plot(x, y, linewidth=2, label=legend if legend else None)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    if legend: plt.legend()
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()

if not os.path.exists(TRAJ_CSV):
    raise FileNotFoundError("traj.csv not found.")
df = pd.read_csv(TRAJ_CSV)
t,x,y,vx,vy,ux,uy = (df[c].to_numpy() for c in ["t","x","y","vx","vy","ux","uy"])

speed = np.sqrt(vx**2 + vy**2)
ax, ay = finite_diff(vx,t), finite_diff(vy,t)
accel  = np.sqrt(ax**2 + ay**2)
u_mag  = np.sqrt(ux**2 + uy**2)
dux, duy = finite_diff(ux,t), finite_diff(uy,t)
udot   = np.sqrt(dux**2 + duy**2)
curv   = curvature(x,y,t)
margin = safety_margin(x,y,OBSTACLES)
Wx, Wy = wind_field(x,y,t)
vdotW  = vx*Wx + vy*Wy
energy_cum  = cumtrapz(u_mag**2, t)
pathlen_cum = cumtrapz(speed, t)

pd.DataFrame({
    "t":t,"x":x,"y":y,"vx":vx,"vy":vy,"speed":speed,"ax":ax,"ay":ay,"accel":accel,
    "ux":ux,"uy":uy,"u_mag":u_mag,"dux_dt":dux,"duy_dt":duy,"udot_mag":udot,
    "curvature":curv,"safety_margin":margin,"Wx":Wx,"Wy":Wy,"v_dot_W":vdotW,
    "energy_cum":energy_cum,"pathlen_cum":pathlen_cum
}).to_csv("metrics_extended.csv", index=False)

# 轨迹（时间线宽渐变）
lw = np.linspace(1.5, 3.0, len(x))
plt.figure(figsize=(7,4.5), dpi=140)
for i in range(1,len(x)):
    plt.plot(x[i-1:i+1], y[i-1:i+1], linewidth=lw[i])
for (cx,cy,r) in OBSTACLES:
    circ = plt.Circle((cx,cy), r, alpha=0.15)
    plt.gca().add_patch(circ); plt.plot(cx,cy,marker="x",linestyle="None")
plt.xlabel("x (m)"); plt.ylabel("y (m)"); plt.title("UAV Trajectory (time-progressive line width)")
plt.grid(True, linestyle="--", alpha=0.3); plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout(); plt.savefig("fig_traj_density.png", bbox_inches="tight"); plt.close()

save_line(t, speed, "t (normalized)", "Speed (m/s)", "UAV Speed vs Time", "fig_speed_time.png", legend="|v|")
save_line(t, accel, "t (normalized)", "Acceleration (m/s²)", "UAV Acceleration vs Time", "fig_accel_time.png", legend="|a|")

plt.figure(figsize=(7,4.5), dpi=140)
plt.plot(t, ux, linewidth=1.8, label="u_x")
plt.plot(t, uy, linewidth=1.8, label="u_y")
plt.plot(t, u_mag, linewidth=1.8, linestyle="--", label="|u|")
plt.xlabel("t (normalized)"); plt.ylabel("Control Input"); plt.title("Control Inputs vs Time")
plt.grid(True, linestyle="--", alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig("fig_control_time.png", bbox_inches="tight"); plt.close()

save_line(t, udot, "t (normalized)", "|du/dt|", "Control Rate vs Time", "fig_control_rate_time.png")
save_line(t, energy_cum, "t (normalized)", "∫||u||² dt", "Cumulative Control Energy", "fig_energy_cum.png")
save_line(t, pathlen_cum, "t (normalized)", "∫||v|| dt (m)", "Cumulative Path Length", "fig_pathlen_cum.png")
save_line(t, curv, "t (normalized)", "Curvature κ", "Curvature vs Time", "fig_curvature_time.png")
save_line(t, margin, "t (normalized)", "Safety Margin (m)", "Distance to Nearest Obstacle Boundary", "fig_safetymargin_time.png")

plt.figure(figsize=(7,4.5), dpi=140)
plt.plot(vx, vy, linewidth=1.8)
plt.xlabel("v_x (m/s)"); plt.ylabel("v_y (m/s)"); plt.title("Phase Portrait: v_x vs v_y")
plt.grid(True, linestyle="--", alpha=0.3); plt.tight_layout(); plt.savefig("fig_phase_vxvy.png", bbox_inches="tight"); plt.close()

plt.figure(figsize=(7,4.5), dpi=140)
plt.plot(ux, uy, linewidth=1.8)
plt.xlabel("u_x"); plt.ylabel("u_y"); plt.title("Control Phase: u_x vs u_y")
plt.grid(True, linestyle="--", alpha=0.3); plt.tight_layout(); plt.savefig("fig_phase_uxuy.png", bbox_inches="tight"); plt.close()

heading = np.degrees(np.arctan2(vy, vx))
save_line(t, heading, "t (normalized)", "Heading ψ (deg)", "Heading Angle vs Time", "fig_heading_time.png")
save_line(t, vdotW, "t (normalized)", "v·W", "Alignment between Velocity and Wind", "fig_vdotW_time.png")
print("[OK] All figures + metrics_extended.csv are saved.")
