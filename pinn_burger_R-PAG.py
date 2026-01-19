import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.integrate import odeint

# --- 1. 全局配置 ---
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 物理参数
NU = 0.01 / np.pi

# ==========================================
# 2. 数值真值生成器 (用于误差分析)
# ==========================================
class BurgersExact:
    def __init__(self, nu):
        self.nu = nu
        self.num_x = 1000
        self.x = np.linspace(-1, 1, self.num_x)
        self.dx = self.x[1] - self.x[0]

    def solve(self, t_target):
        """使用有限差分法 + odeint 计算 t_target 时刻的精确解"""
        # 初始条件 u(0,x) = -sin(pi*x)
        u0 = -np.sin(np.pi * self.x)

        # 定义 ODE 系统: du/dt = RHS
        def deriv(u, t):
            dudt = np.zeros_like(u)
            # 边界条件 u(-1)=0, u(1)=0 (dudt在边界为0)
            
            # 内部点: 粘性项 (nu * u_xx) + 对流项 (-u * u_x)
            # 使用中心差分
            u_xx = (u[2:] - 2*u[1:-1] + u[:-2]) / self.dx**2
            u_x  = (u[2:] - u[:-2]) / (2*self.dx)
            
            dudt[1:-1] = self.nu * u_xx - u[1:-1] * u_x
            return dudt

        # 积分到 t_target
        t_eval = np.linspace(0, t_target, 100)
        sol = odeint(deriv, u0, t_eval)
        
        return self.x, sol[-1]

# ==========================================
# 3. 网络架构 (保持与 Baseline 一致)
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # 5 层全连接网络，每层 50 个神经元，Tanh 激活
        self.layer = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )
    
    def forward(self, t, x):
        return self.layer(torch.cat([t, x], dim=1))

# ==========================================
# 4. R-PAG 求解器 (V5.0: 带 Warm-up 和 权重修正)
# ==========================================
class BurgersRPAGSolver:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # GCond: 历史梯度缓冲区 (仅平滑组使用)
        self.V_smooth = {} 
        self.beta = 0.9 # 动量系数
        for name, param in model.named_parameters():
            self.V_smooth[name] = torch.zeros_like(param.data)

    def get_gradients(self, loss):
        """计算梯度但不更新"""
        grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True, allow_unused=True)
        grad_dict = {}
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            if grad is not None:
                grad_dict[name] = grad
            else:
                grad_dict[name] = torch.zeros_like(param)
        return grad_dict

    def asymmetric_projection(self, g_low, g_high):
        """R-PAG 核心: 冲突投影"""
        g_low_flat = torch.cat([g.flatten() for g in g_low.values()])
        g_high_flat = torch.cat([g.flatten() for g in g_high.values()])
        
        dot = torch.dot(g_low_flat, g_high_flat)
        
        if dot < 0: # 冲突检测
            norm_sq = torch.dot(g_high_flat, g_high_flat) + 1e-8
            proj_coeff = dot / norm_sq
            
            g_low_new = {}
            for name in g_low:
                g_low_new[name] = g_low[name] - proj_coeff * g_high[name]
            return g_low_new, True # 发生投影
        
        return g_low, False # 无冲突

    def step(self, t, x, t_bc, x_left, x_right, x_ic, use_rpag=True):
        self.optimizer.zero_grad()
        
        # --- A. PDE 残差计算 ---
        u = self.model(t, x)
        u_t = torch.autograd.grad(u, t, torch.ones_like(t), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(x), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(x), create_graph=True)[0]
        
        res_pde = u_t + u * u_x - NU * u_xx
        res_sq = res_pde.pow(2)
        
        # --- B. BC & IC Loss ---
        u_left = self.model(t_bc, x_left)
        u_right = self.model(t_bc, x_right)
        L_bc = u_left.pow(2).mean() + u_right.pow(2).mean()
        
        t_0 = torch.zeros_like(x_ic)
        u_init = self.model(t_0, x_ic)
        u_exact_ic = -torch.sin(torch.pi * x_ic)
        L_ic = (u_init - u_exact_ic).pow(2).mean()

        is_conflict = False
        
        # === 分支 1: Warm-up 阶段 (像 Baseline 一样运行) ===
        if not use_rpag:
            # 直接相加，不加权，保证起点一致
            L_pde_mean = res_sq.mean()
            loss = L_pde_mean + L_bc + L_ic
            loss.backward()
            self.optimizer.step()
            return loss.item(), False

        # === 分支 2: R-PAG 阶段 (精修激波) ===
        else:
            # 1. 动态分组
            res_abs = torch.sqrt(res_sq).detach()
            threshold = torch.quantile(res_abs, 0.90) # Top 10%
            
            mask_shock = (res_abs >= threshold).float()
            mask_smooth = 1.0 - mask_shock
            
            L_shock = torch.sum(res_sq * mask_shock) / (torch.sum(mask_shock) + 1e-8)
            L_smooth = torch.sum(res_sq * mask_smooth) / (torch.sum(mask_smooth) + 1e-8)
            
            # 2. 梯度计算 (权重修正：不再使用 20.0/100.0，改用温和的 2.0/5.0)
            g_shock = self.get_gradients(L_shock * 2.0)
            g_constraints = self.get_gradients((L_bc + L_ic) * 5.0)
            
            g_high = {}
            for name in g_shock:
                g_high[name] = g_shock[name] + g_constraints[name]
            
            g_smooth_raw = self.get_gradients(L_smooth * 1.0)
            
            # 3. GCond (历史梯度平滑)
            g_smooth_stable = {}
            for name in g_smooth_raw:
                self.V_smooth[name] = self.beta * self.V_smooth[name] + (1 - self.beta) * g_smooth_raw[name]
                g_smooth_stable[name] = self.V_smooth[name]
            
            # 4. 投影
            g_smooth_final, is_conflict = self.asymmetric_projection(g_smooth_stable, g_high)
            
            # 5. 更新
            for name, param in self.model.named_parameters():
                param.grad = g_high[name] + g_smooth_final[name]
            
            self.optimizer.step()
            
            return (L_shock + L_smooth + L_bc + L_ic).item(), is_conflict

# ==========================================
# 5. 主程序与评估工具
# ==========================================
def evaluate_and_plot(model, exact_solver, t_val=0.5):
    """计算误差并绘图"""
    # 1. 获取真值
    print(f"Calculating numerical reference for t={t_val}...")
    x_true, u_true = exact_solver.solve(t_val)
    
    # 2. 获取预测值
    x_tensor = torch.tensor(x_true.reshape(-1, 1), dtype=torch.float32).to(device)
    t_tensor = torch.full((len(x_true), 1), t_val, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(t_tensor, x_tensor).cpu().numpy().flatten()
    
    # 3. 计算相对 L2 误差
    error_l2 = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
    abs_error = np.abs(u_true - u_pred)
    
    print(f"Time: {t_val}s | Relative L2 Error: {error_l2:.4e}")

    # 4. 绘图 (双子图)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 解的对比
    axs[0].plot(x_true, u_true, 'k-', lw=2, label='Numerical Exact')
    axs[0].plot(x_true, u_pred, 'r--', lw=2, label='R-PAG Prediction')
    axs[0].set_title(f"Solution Comparison (t={t_val})\nRel L2 Error: {error_l2:.4e}", color='blue', fontsize=14)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel(f"u(t={t_val}, x)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # 右图: 绝对误差分布
    axs[1].plot(x_true, abs_error, 'b-', lw=1)
    axs[1].fill_between(x_true, abs_error, color='blue', alpha=0.1)
    axs[1].set_yscale('log')
    axs[1].set_title("Absolute Error Distribution", fontsize=14)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("|u_exact - u_pred|")
    axs[1].grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return error_l2

def main():
    # 初始化
    model = PINN().to(device)
    solver = BurgersRPAGSolver(model, lr=1e-3)
    exact_solver = BurgersExact(NU)
    
    epochs = 15000
    loss_history = []
    
    print(f">>> Start Training (Epochs: {epochs}) <<<")
    
    # --- 时间统计开始 ---
    start_time = time.time()
    
    for ep in range(epochs):
        # 1. 每一轮重新采样内部点
        t = torch.rand(2000, 1, requires_grad=True, device=device)
        x = torch.rand(2000, 1, requires_grad=True, device=device) * 2 - 1
        
        # 2. 边界点
        t_bc = torch.rand(500, 1, device=device)
        x_l = -torch.ones(500, 1, device=device)
        x_r = torch.ones(500, 1, device=device)
        
        # 3. 初始点
        x_ic = torch.rand(1000, 1, device=device) * 2 - 1
        
        # 4. 训练步 (Warm-up 逻辑)
        # 前 20% (3000轮) 用普通模式，之后开启 R-PAG
        if ep < epochs * 0.2:
            use_rpag = False
        else:
            use_rpag = True
            
        loss, is_proj = solver.step(t, x, t_bc, x_l, x_r, x_ic, use_rpag=use_rpag)
        loss_history.append(loss)
        
        if ep % 1000 == 0:
            mode_str = "R-PAG" if use_rpag else "WarmUp"
            proj_str = "YES" if is_proj else "NO"
            print(f"Ep {ep:05d} | Mode: {mode_str} | Loss: {loss:.6f} | Proj: {proj_str}")

    # --- 时间统计结束 ---
    end_time = time.time()
    training_time = end_time - start_time
    
    print("\n" + "="*40)
    print(f"Training Completed.")
    print(f"Total Time: {training_time:.2f} seconds")
    print("="*40 + "\n")
    
    # --- 误差分析与可视化 ---
    final_error = evaluate_and_plot(model, exact_solver, t_val=0.5)
    
    # 绘制 Loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title("Training Loss (Warm-up -> R-PAG)")
    plt.xlabel("Epoch")
    plt.show()

if __name__ == "__main__":
    main()