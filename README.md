<!-- ===== UAV-PINN README Visual Section ===== -->

<div align="center">

# 🛰️ Physics-Informed UAV Trajectory Planning  
*A Unified Learning–Physics Framework for Safe, Smooth, and Energy-Efficient Flight in Dynamic Environments*  

<!-- ===== Main Animation ===== -->
<img src="fig/traj_PINN.gif" width="75%" alt="PINN UAV trajectory animation"/>

<p><em>Figure 1. Trajectory evolution of the proposed Physics-Informed Neural Network (PINN) model under dynamic wind fields.</em></p>

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![Stars](https://img.shields.io/github/stars/zhuoyongshi/uav-pinn-planning?style=social)](#)
[![arXiv](https://img.shields.io/badge/arXiv-2501.21874-b31b1b.svg)](https://arxiv.org/abs/2510.21874)

---

### ✈️ Comparative Flight Dynamics

<table>
<tr>
<td align="center"><b>A* Algorithm</b><br><img src="fig/traj_A.gif" width="95%"/></td>
<td align="center"><b>Kino-RRT*</b><br><img src="fig/traj_RRT.gif" width="95%"/></td>
</tr>
</table>

<p><em>Figure 2. Baseline trajectories of A* (left) and Kino-RRT* (right) in the same environment.  
Both rely on discrete sampling and lack dynamic smoothness compared to the PINN approach.</em></p>

---

### 🧠 Evolution of Model Parameters

<table>
<tr>
<td align="center"><img src="fig/figure_mosaic.png" width="90%"/><br><em>(a) PINN Loss Convergence</em></td>
<td align="center"><img src="fig/figure_mosaic_A.png" width="90%"/><br><em>(b) A* Energy Profile</em></td>
<td align="center"><img src="fig/figure_mosaic_rrt.png" width="90%"/><br><em>(c) RRT* Energy Profile</em></td>
</tr>
</table>

<p><em>Figure 3. Training and control-energy evolution across different algorithms.  
The PINN demonstrates smoother convergence and lower energy oscillation amplitude.</em></p>

---

### 📊 Quantitative Comparison

<img src="fig/bars_all_metrics.png" width="80%"/>

<p><em>Figure 4. Quantitative evaluation of flight performance:  
energy consumption, smoothness, and risk proximity.  
The physics-informed model achieves the best trade-off across all metrics.</em></p>

</div>
