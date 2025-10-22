# astar_wind_2d.py
# Wind-aware A* on 2D grid; outputs traj.csv (t,x,y,vx,vy,ux,uy) and traj.jpg
import math, os, heapq, time
import numpy as np
import matplotlib.pyplot as plt

class Cfg:
    # Start/Goal (meters) — keep identical to your PINN
    x0, y0 = 0.0, 0.0
    xT, yT = 10.0, 6.0

    # Wind field params (identical form to PINN)
    W_Ax = 1.2
    W_Ay = 0.8

    # Obstacles: list of (cx, cy, r)
    obstacles = [(5.0,3.0,1.2), (7.0,2.0,1.0), (3.0,2.0,0.8)]

    # Grid & motion
    ds = 0.2              # grid step (m)
    v_cmd = 2.0           # commanded cruise speed (m/s)  -> dt = ds / v_cmd
    allow_diag = True     # 8-connected grid

    # Cost weights (aligned with PINN objective)
    w_t = 1.0
    w_e = 1.0
    w_r = 15.0            # obstacle risk weight (same order as PINN gamma)
    alpha_soft = 10.0     # softplus slope for inside-penalty
    eps = 1e-4

    # Goal tolerance
    goal_radius = 0.3     # reach this radius around goal -> success

    # Bounding box for search (pad around [start, goal])
    x_min, x_max = -1.0, 12.0
    y_min, y_max = -1.0,  8.0

    # Export
    out_csv = "traj.csv"
    out_fig = "traj.jpg"

cfg = Cfg()

def wind_field(x, y, t):
    Wx = cfg.W_Ax * np.sin(np.pi * y / 6.0) * np.cos(2*np.pi*t)
    Wy = cfg.W_Ay * np.sin(np.pi * x / 10.0) * np.sin(2*np.pi*t)
    return Wx, Wy

def risk_phi(x, y):
    # phi(d) = 1/(d^2+eps) + softplus(-alpha d)
    total = 0.0
    for (cx, cy, r) in cfg.obstacles:
        d = np.sqrt((x-cx)**2 + (y-cy)**2 + cfg.eps) - r
        total += 1.0 / (d*d + cfg.eps) + np.log1p(np.exp(-cfg.alpha_soft * d))
    return total

def segment_hits_any_circle(x1,y1, x2,y2):
    # Check if segment intersects any circle obstacle
    for (cx,cy,r) in cfg.obstacles:
        # distance from circle center to segment
        vx, vy = x2-x1, y2-y1
        wx, wy = cx-x1, cy-y1
        c1 = vx*wx + vy*wy
        c2 = vx*vx + vy*vy + 1e-12
        u  = np.clip(c1/c2, 0.0, 1.0)
        px, py = x1 + u*vx, y1 + u*vy
        if (px-cx)**2 + (py-cy)**2 <= r*r:
            return True
    return False

def astar():
    ds = cfg.ds
    # grid index mapping
    def to_idx(x, y):
        i = int(round((x - cfg.x_min) / ds))
        j = int(round((y - cfg.y_min) / ds))
        return i, j
    def to_xy(i, j):
        x = cfg.x_min + i*ds
        y = cfg.y_min + j*ds
        return x, y

    # neighbors
    if cfg.allow_diag:
        nbr = [(1,0,1.0),(0,1,1.0),(-1,0,1.0),(0,-1,1.0),
               (1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),
               (-1,1,math.sqrt(2)),(-1,-1,math.sqrt(2))]
    else:
        nbr = [(1,0,1.0),(0,1,1.0),(-1,0,1.0),(0,-1,1.0)]

    # bounds
    imax = int(math.floor((cfg.x_max - cfg.x_min)/ds))
    jmax = int(math.floor((cfg.y_max - cfg.y_min)/ds))

    # start/goal index
    is_, js_ = to_idx(cfg.x0, cfg.y0)
    ig,  jg  = to_idx(cfg.xT, cfg.yT)

    # heuristic: geometric time lower bound
    def h(i,j):
        x,y = to_xy(i,j)
        dg = math.hypot(cfg.xT-x, cfg.yT-y)
        return cfg.w_t * (dg / cfg.v_cmd)

    # open set (f,i,j,t, parent_key)
    t0 = 0.0
    key0 = (is_,js_)
    openpq = []
    heapq.heappush(openpq, (h(is_,js_), 0.0, is_, js_, t0, None))
    came = {}  # (i,j) -> (parent_key, t, g)
    came[key0] = (None, t0, 0.0)

    def in_bounds(i,j):
        return 0 <= i <= imax and 0 <= j <= jmax

    t_start = time.time()
    goal_key = None

    while openpq:
        f, g_curr, i, j, t_curr, parent = heapq.heappop(openpq)
        # early goal check by continuous distance
        x,y = to_xy(i,j)
        if math.hypot(x-cfg.xT, y-cfg.yT) <= cfg.goal_radius:
            goal_key = (i,j)
            came[goal_key] = (parent, t_curr, g_curr)
            break

        # If not the best entry (inconsistency), skip
        if (i,j) not in came: 
            continue
        prev_parent, prev_t, prev_g = came[(i,j)]
        if abs(prev_t - t_curr) > 1e-9 or abs(prev_g - g_curr) > 1e-9:
            continue

        for di, dj, step_factor in nbr:
            ni, nj = i+di, j+dj
            if not in_bounds(ni,nj): 
                continue
            x2,y2 = to_xy(ni,nj)
            # segment collision
            if segment_hits_any_circle(x,y, x2,y2):
                continue
            # edge metrics
            ds_eff = cfg.ds * step_factor
            dt = ds_eff / cfg.v_cmd
            # energy proxy: ||v_cmd ê - W||^2 * dt
            ex, ey = (x2-x)/ds_eff, (y2-y)/ds_eff  # unit dir
            Wx, Wy = wind_field(x, y, t_curr)
            u_proxy_sq = (cfg.v_cmd*ex - Wx)**2 + (cfg.v_cmd*ey - Wy)**2
            # risk at start point (or midpoint)
            r = risk_phi(x, y)
            cost = cfg.w_t*dt + cfg.w_e*u_proxy_sq*dt + cfg.w_r*r
            g_new = g_curr + cost
            t_new = t_curr + dt
            keyn = (ni,nj)
            old = came.get(keyn, None)
            if old is None or g_new < old[2] - 1e-12:
                came[keyn] = ((i,j), t_new, g_new)
                heapq.heappush(openpq, (g_new + h(ni,nj), g_new, ni, nj, t_new, (i,j)))

    if goal_key is None:
        raise RuntimeError("A* failed to find a path")

    # reconstruct
    path = []
    key = goal_key
    while key is not None:
        (par, tval, gval) = came[key]
        x,y = to_xy(*key)
        path.append((tval, x, y))
        key = par
    path.reverse()

    # densify (optional): already time-parametrized with dt = ds/v_cmd at steps
    path = np.array(path)  # (N,3): t,x,y
    return path, time.time()-t_start

def export(path):
    # compute vx,vy via finite difference; u proxy via v_cmd*ê - W
    t, x, y = path[:,0], path[:,1], path[:,2]
    dt = np.diff(t); dt = np.append(dt, dt[-1] if dt.size>0 else 1e-2)
    vx = np.gradient(x, t, edge_order=2)
    vy = np.gradient(y, t, edge_order=2)

    # control proxy
    ux = np.zeros_like(x); uy = np.zeros_like(y)
    for k in range(len(x)-1):
        dx, dy = x[k+1]-x[k], y[k+1]-y[k]
        ds = max(np.hypot(dx,dy), 1e-9)
        ex, ey = dx/ds, dy/ds
        Wx, Wy = wind_field(x[k], y[k], t[k])
        ux[k] = cfg.v_cmd*ex - Wx
        uy[k] = cfg.v_cmd*ey - Wy
    # last point: reuse previous
    ux[-1], uy[-1] = ux[-2], uy[-2]

    data = np.stack([t,x,y,vx,vy,ux,uy], axis=1)
    np.savetxt(cfg.out_csv, data, delimiter=",", header="t,x,y,vx,vy,ux,uy", comments="")
    print(f"[OK] Saved CSV: {cfg.out_csv}")

    # Plot
    fig, ax = plt.subplots(figsize=(7,4.5), dpi=140)
    ax.plot(x, y, lw=2, label="A*")
    ax.scatter([cfg.x0, cfg.xT], [cfg.y0, cfg.yT], c=["green","red"], s=60, zorder=5, label="Start/Goal")
    for (cx, cy, r) in cfg.obstacles:
        circ = plt.Circle((cx,cy), r, color="gray", alpha=0.25)
        ax.add_patch(circ); ax.plot(cx, cy, "kx", ms=6)
    # wind at mid-time
    Xg, Yg = np.meshgrid(np.linspace(min(x.min(),0)-1, max(x.max(),cfg.xT)+1, 20),
                         np.linspace(min(y.min(),0)-1, max(y.max(),cfg.yT)+1, 14))
    Tg = np.full_like(Xg, 0.5*(t[0]+t[-1]))
    Wx, Wy = wind_field(Xg, Yg, Tg)
    ax.quiver(Xg, Yg, Wx, Wy, alpha=0.35, pivot="mid", angles="xy", scale_units="xy", scale=5.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Wind-aware A* Trajectory")
    ax.grid(True, ls="--", alpha=0.3); ax.legend(loc="best")
    plt.tight_layout(); plt.savefig(cfg.out_fig, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] Saved Figure: {cfg.out_fig}")

def main():
    path, tcost = astar()
    print(f"[A*] nodes={len(path)}  wall_time={tcost:.3f}s")
    export(path)

if __name__ == "__main__":
    main()
