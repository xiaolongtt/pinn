import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import newton

# --- 1. 全局配置 ---
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 物理参数 (SOD 标准工况)
GAMMA = 1.4

# ==========================================
# 2. 网络架构: Multi-Output SIREN
# ==========================================
class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = w0
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_features if self.is_first else np.sqrt(6 / self.in_features) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class EulerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输出: rho(密度), u(速度), p(压力)
        self.net = nn.Sequential(
            SIRENLayer(2, 64, w0=30.0, is_first=True),
            SIRENLayer(64, 64, w0=1.0),
            SIRENLayer(64, 64, w0=1.0),
            SIRENLayer(64, 64, w0=1.0),
            SIRENLayer(64, 64, w0=1.0),
            nn.Linear(64, 3) 
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 真实解 (SOD Riemann Solver)
# ==========================================
class SodExact:
    def __init__(self):
        self.gamma = 1.4
        self.mu = np.sqrt((self.gamma-1)/(self.gamma+1))
        # 初始条件
        self.rho_l, self.p_l, self.u_l = 1.0, 1.0, 0.0
        self.rho_r, self.p_r, self.u_r = 0.125, 0.1, 0.0

    def solve(self, t):
        # 求解中心压力 p_star
        def f(p, rho, p_init):
            if p >= p_init:
                return (p - p_init) * np.sqrt((1 - self.mu**2) / (rho * (p + self.mu**2 * p_init)))
            else:
                return (2 * np.sqrt(self.gamma * p_init / rho) / (self.gamma - 1)) * ((p / p_init)**((self.gamma - 1) / (2 * self.gamma)) - 1)

        def p_star_func(p):
            return f(p, self.rho_l, self.p_l) + f(p, self.rho_r, self.p_r) + (self.u_r - self.u_l)

        p_star = newton(p_star_func, 0.5)
        
        # 计算速度 u_star
        u_star = 0.5 * (self.u_l + self.u_r) + 0.5 * (f(p_star, self.rho_r, self.p_r) - f(p_star, self.rho_l, self.p_l))
        
        x = np.linspace(0, 1, 1000)
        rho_res, u_res, p_res = [], [], []

        # 计算声速
        c_l = np.sqrt(self.gamma * self.p_l / self.rho_l)
        c_r = np.sqrt(self.gamma * self.p_r / self.rho_r)

        for xi in x:
            # 波速位置
            x_t = (xi - 0.5) / (t + 1e-10) # 偏移中心 0.5

            if x_t < self.u_l - c_l: # Region 1 (Left State)
                rho_res.append(self.rho_l); u_res.append(self.u_l); p_res.append(self.p_l)
            
            elif x_t < u_star * (1 if p_star > self.p_l else -1): # 简化判断，这里只针对典型SOD情况 (左稀疏，右激波)
                 # Rarefaction Wave (Left)
                 if p_star <= self.p_l: # 只有左侧压力大时才会形成左稀疏波
                     if x_t < u_star - np.sqrt(self.gamma*p_star/((p_star/self.p_l)**(1/self.gamma)*self.rho_l)): # Head of rarefaction
                         # Inside Rarefaction Fan
                         u_fan = 2/(self.gamma+1) * (c_l + (self.gamma-1)/2 * self.u_l + x_t)
                         c_fan = 2/(self.gamma+1) * (c_l + (self.gamma-1)/2 * (self.u_l - x_t))
                         rho_fan = self.rho_l * (c_fan/c_l)**(2/(self.gamma-1))
                         p_fan = self.p_l * (rho_fan/self.rho_l)**self.gamma
                         rho_res.append(rho_fan); u_res.append(u_fan); p_res.append(p_fan)
                     else: # Region 3 (Star Region Left)
                         rho_star_l = self.rho_l * (p_star/self.p_l)**(1/self.gamma)
                         rho_res.append(rho_star_l); u_res.append(u_star); p_res.append(p_star)
                 else: # Shock Left (Not SOD case usually)
                     pass

            elif x_t < u_star: # Region 3 (Contact Discontinuity Left)
                 rho_star_l = self.rho_l * (p_star/self.p_l)**(1/self.gamma)
                 rho_res.append(rho_star_l); u_res.append(u_star); p_res.append(p_star)
            
            elif x_t < u_star + (self.p_r/self.rho_r)**0.5 * ((self.gamma+1)/(2*self.gamma)*p_star/self.p_r + (self.gamma-1)/(2*self.gamma))**0.5: # Region 4 (Star Region Right) -- Check shock speed
                 # Shock speed calculation
                 shock_speed = u_star + np.sqrt(self.gamma*self.p_r/self.rho_r * ((self.gamma+1)/(2*self.gamma)*p_star/self.p_r + (self.gamma-1)/(2*self.gamma)))
                 # Correct shock location check
                 s_shock = self.u_r + c_r * np.sqrt((self.gamma+1)/(2*self.gamma)*p_star/self.p_r + (self.gamma-1)/(2*self.gamma))
                 
                 if x_t < u_star: # Contact
                     rho_star_r = self.rho_r * ((p_star + self.mu**2 * self.p_r) / (self.p_r + self.mu**2 * p_star))
                     rho_res.append(rho_star_r); u_res.append(u_star); p_res.append(p_star)
                 elif x_t < s_shock: # Region 4
                      rho_star_r = self.rho_r * ((p_star + self.mu**2 * self.p_r) / (self.p_r + self.mu**2 * p_star))
                      rho_res.append(rho_star_r); u_res.append(u_star); p_res.append(p_star)
                 else: # Region 5
                      rho_res.append(self.rho_r); u_res.append(self.u_r); p_res.append(self.p_r)
            else: # Region 5 (Right State)
                 rho_res.append(self.rho_r); u_res.append(self.u_r); p_res.append(self.p_r)

        return x, np.array(rho_res), np.array(u_res), np.array(p_res)

# ==========================================
# 4. 混合采样器
# ==========================================
class HybridSampler:
    def __init__(self, model, n_total=2000):
        self.model = model
        self.n_total = n_total
    
    def resample(self, epsilon_visc):
        # 50% 均匀 + 50% 残差自适应
        n_uniform = int(self.n_total * 0.5)
        n_adaptive = self.n_total - n_uniform
        
        # 均匀采样
        x_u = torch.rand(n_uniform, 1)
        t_u = torch.rand(n_uniform, 1) * 0.2 # t in [0, 0.2]
        X_uniform = torch.cat([x_u, t_u], dim=1).to(device)
        
        # 候选池
        x_c = torch.rand(20000, 1)
        t_c = torch.rand(20000, 1) * 0.2
        X_cand = torch.cat([x_c, t_c], dim=1).to(device)
        X_cand.requires_grad_(True)
        
        # 计算残差 (需要计算 Euler 残差)
        # 这里简化：只计算 continuity 残差作为采样依据，速度快
        out = self.model(X_cand)
        rho, u, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        
        grads_rho = torch.autograd.grad(rho, X_cand, torch.ones_like(rho), 
                                        create_graph=False, retain_graph=True)[0]
        rho_x, rho_t = grads_rho[:, 0:1], grads_rho[:, 1:2]
        
        # 简单的残差代理：rho_t + (rho*u)_x
        # 只需要找梯度大的地方
        grads_u = torch.autograd.grad(u, X_cand, torch.ones_like(u), 
                                      create_graph=False, retain_graph=False)[0]
        u_x = grads_u[:, 0:1]
        
        res = rho_t + rho * u_x + u * rho_x # Continuity
        res_abs = torch.abs(res).detach().cpu().numpy().flatten()
        
        prob = res_abs ** 2
        prob = prob / (prob.sum() + 1e-10)
        
        indices = np.random.choice(len(prob), n_adaptive, replace=False, p=prob)
        X_adaptive = X_cand[indices].detach()
        
        X_final = torch.cat([X_uniform, X_adaptive], dim=0).detach()
        X_final.requires_grad_(True)
        return X_final

# ==========================================
# 5. R-PAG Euler 求解器
# ==========================================
class EulerRPAGSolver:
    def __init__(self, model):
        self.model = model
        self.optim_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim_adam, gamma=0.95)
        
        self.optim_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, history_size=100, max_iter=3000, max_eval=3000,
            line_search_fn="strong_wolfe"
        )
        
        # GCond 历史
        self.V_smooth = {} 
        self.beta = 0.9
        for name, param in model.named_parameters():
            self.V_smooth[name] = torch.zeros_like(param.data)

    def compute_residuals(self, X_pde, eps_visc):
        out = self.model(X_pde)
        rho, u, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        
        # 一阶导
        grads_rho = torch.autograd.grad(rho, X_pde, torch.ones_like(rho), create_graph=True)[0]
        rho_x, rho_t = grads_rho[:, 0:1], grads_rho[:, 1:2]
        
        grads_u = torch.autograd.grad(u, X_pde, torch.ones_like(u), create_graph=True)[0]
        u_x, u_t = grads_u[:, 0:1], grads_u[:, 1:2]
        
        grads_p = torch.autograd.grad(p, X_pde, torch.ones_like(p), create_graph=True)[0]
        p_x, p_t = grads_p[:, 0:1], grads_p[:, 1:2]
        
        # 二阶导 (人工粘性项)
        rho_xx = torch.autograd.grad(rho_x, X_pde, torch.ones_like(rho_x), create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x, X_pde, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        p_xx = torch.autograd.grad(p_x, X_pde, torch.ones_like(p_x), create_graph=True)[0][:, 0:1] # E的粘性近似
        
        # 能量 E = p/(gamma-1) + 0.5*rho*u^2
        E = p / (GAMMA - 1) + 0.5 * rho * u**2
        
        # 1. 连续性方程: rho_t + (rho*u)_x = eps * rho_xx
        res_mass = rho_t + (rho * u_x + u * rho_x) - eps_visc * rho_xx
        
        # 2. 动量方程: (rho*u)_t + (rho*u^2 + p)_x = eps * u_xx
        # 简化非守恒形式: rho*(u_t + u*u_x) + p_x = ...
        res_mom = rho * (u_t + u * u_x) + p_x - eps_visc * u_xx
        
        # 3. 能量方程: E_t + ((E+p)*u)_x = eps * E_xx
        # 简化为压力形式: p_t + u*p_x + gamma*p*u_x = ...
        res_energy = p_t + u * p_x + GAMMA * p * u_x - eps_visc * p_xx

        return res_mass, res_mom, res_energy

    def compute_loss(self, X_pde, X_ic, U_ic, X_bc,U_bc, eps_visc):
        # PDE Residuals
        r1, r2, r3 = self.compute_residuals(X_pde, eps_visc)
        res_sq = r1**2 + r2**2 + r3**2
        
        # 动态分组
        res_abs = torch.sqrt(res_sq).detach()
        threshold = torch.quantile(res_abs, 0.95)
        mask_shock = (res_abs >= threshold).float()
        mask_smooth = 1.0 - mask_shock
        
        L_shock = torch.sum(res_sq * mask_shock) / (torch.sum(mask_shock) + 1e-8)
        L_smooth = torch.sum(res_sq * mask_smooth) / (torch.sum(mask_smooth) + 1e-8)
        
        # IC Loss (t=0)
        out_ic = self.model(X_ic)
        L_ic = torch.mean((out_ic - U_ic)**2)
        
        # BC Loss (x=0 and x=1, Dirichlet)
        # SOD问题边界在0.2s内不会受到波的影响，保持初值
        out_bc = self.model(X_bc)
        L_bc = torch.mean((out_bc - U_bc)**2)
        # 这里 X_bc 需要对应正确的 U_bc (左边是L状态，右边是R状态)
        # 后面主程序会构造正确的 U_bc
        # 假设 U_bc 已经传入正确
        L_bc = torch.mean((out_bc - U_bc)**2) # 这里的U_bc需要包含rho, u, p
        
        return L_shock, L_smooth, L_ic, L_bc

    def step_adam_rpag(self, X_pde, X_ic, U_ic, X_bc, U_bc, eps_visc):
        self.optim_adam.zero_grad()
        L_shock, L_smooth, L_ic, L_bc = self.compute_loss(X_pde, X_ic, U_ic, X_bc, U_bc, eps_visc)
        
        # 梯度权重
        g_shock = torch.autograd.grad(L_shock * 20.0, self.model.parameters(), retain_graph=True, allow_unused=True)
        g_smooth = torch.autograd.grad(L_smooth * 1.0, self.model.parameters(), retain_graph=True, allow_unused=True)
        g_icbc = torch.autograd.grad((L_ic + L_bc) * 100.0, self.model.parameters(), retain_graph=True, allow_unused=True)
        
        # R-PAG 投影
        params = list(self.model.named_parameters())
        g_sv, g_hv = [], []
        g_smooth_dict = {}
        
        for idx, (name, param) in enumerate(params):
            gs = g_smooth[idx] if g_smooth[idx] is not None else torch.zeros_like(param)
            gh = g_shock[idx] if g_shock[idx] is not None else torch.zeros_like(param)
            
            self.V_smooth[name] = self.beta * self.V_smooth[name] + (1 - self.beta) * gs
            g_curr = self.V_smooth[name]
            g_smooth_dict[name] = g_curr
            
            g_sv.append(gh.flatten())
            g_hv.append(g_curr.flatten())
            
        dot = torch.dot(torch.cat(g_sv), torch.cat(g_hv))
        
        if dot < 0:
            norm = torch.dot(torch.cat(g_sv), torch.cat(g_sv)) + 1e-8
            coeff = dot / norm
            for idx, (name, param) in enumerate(params):
                gh = g_shock[idx] if g_shock[idx] is not None else torch.zeros_like(param)
                gi = g_icbc[idx] if g_icbc[idx] is not None else torch.zeros_like(param)
                gs = g_smooth_dict[name] - coeff * gh
                param.grad = gh + gs + gi
        else:
            for idx, (name, param) in enumerate(params):
                gh = g_shock[idx] if g_shock[idx] is not None else torch.zeros_like(param)
                gi = g_icbc[idx] if g_icbc[idx] is not None else torch.zeros_like(param)
                param.grad = gh + g_smooth_dict[name] + gi
                
        self.optim_adam.step()
        return (L_shock + L_smooth + L_ic + L_bc).item()

    def step_lbfgs(self, X_pde, X_ic, U_ic, X_bc, U_bc, eps_visc):
        def closure():
            self.optim_lbfgs.zero_grad()
            L_shock, L_smooth, L_ic, L_bc = self.compute_loss(X_pde, X_ic, U_ic, X_bc, U_bc, eps_visc)
            loss = 100.0 * (L_ic + L_bc) + 20.0 * L_shock + 1.0 * L_smooth
            loss.backward()
            print(f"\rL-BFGS Loss: {loss.item():.5f}", end="")
            return loss
        self.optim_lbfgs.step(closure)

# ==========================================
# 6. 主程序
# ==========================================
def main():
    model = EulerPINN().to(device)
    solver = EulerRPAGSolver(model)
    sampler = HybridSampler(model, n_total=4000)
    
    # --- 构造 IC (t=0) ---
    N_ic = 2000
    x_ic = torch.rand(N_ic, 1)
    t_ic = torch.zeros(N_ic, 1)
    X_ic = torch.cat([x_ic, t_ic], dim=1).to(device)
    
    # IC Values (SOD)
    # x <= 0.5: rho=1, u=0, p=1
    # x > 0.5: rho=0.125, u=0, p=0.1
    rho_ic = torch.ones(N_ic, 1)
    rho_ic[x_ic > 0.5] = 0.125
    u_ic = torch.zeros(N_ic, 1)
    p_ic = torch.ones(N_ic, 1)
    p_ic[x_ic > 0.5] = 0.1
    U_ic = torch.cat([rho_ic, u_ic, p_ic], dim=1).to(device)
    
    # --- 构造 BC (x=0, x=1) ---
    N_bc = 500
    t_bc = torch.rand(N_bc, 1) * 0.2
    
    # Left BC (x=0) -> State L
    x_l = torch.zeros(N_bc, 1)
    X_bc_l = torch.cat([x_l, t_bc], dim=1).to(device)
    U_bc_l = torch.cat([torch.ones(N_bc, 1), torch.zeros(N_bc, 1), torch.ones(N_bc, 1)], dim=1).to(device)
    
    # Right BC (x=1) -> State R
    x_r = torch.ones(N_bc, 1)
    X_bc_r = torch.cat([x_r, t_bc], dim=1).to(device)
    U_bc_r = torch.cat([torch.ones(N_bc, 1)*0.125, torch.zeros(N_bc, 1), torch.ones(N_bc, 1)*0.1], dim=1).to(device)
    
    X_bc = torch.cat([X_bc_l, X_bc_r], dim=0)
    U_bc = torch.cat([U_bc_l, U_bc_r], dim=0) # U_bc 包含正确的左右边界值
    
    # --- 粘性延拓训练 (Curriculum) ---
    # Euler 是无粘的 (nu=0)，直接练很难。我们从 nu=0.01 开始，减小到 0.0001 (近似无粘)
    viscosity_schedule = [0.01, 0.005, 0.001]
    epoch_schedule = [3000, 3000, 5000]
    
    print(">>> Start SOD Shock Tube Training <<<")
    
    for stage, (visc, eps) in enumerate(zip(viscosity_schedule, epoch_schedule)):
        print(f"\nStage {stage+1}: Artificial Viscosity = {visc}")
        X_pde = sampler.resample(visc)
        
        for ep in range(eps):
            if ep % 500 == 0 and ep > 0:
                X_pde = sampler.resample(visc)
                
            loss = solver.step_adam_rpag(X_pde, X_ic, U_ic, X_bc, U_bc, visc)
            
            if ep % 1000 == 0:
                print(f"Ep {ep:05d} | Loss: {loss:.4f}")
        
        solver.scheduler.step()

    print("\n>>> Final Polishing with L-BFGS (Viscosity -> 0) <<<")
    # 最后一步假设极小粘性
    X_pde = sampler.resample(1e-5)
    solver.step_lbfgs(X_pde, X_ic, U_ic, X_bc, U_bc, 1e-5)
    
    # ==========================
    # 验证与绘图
    # ==========================
    print("\nGenerating Plots...")
    exact_solver = SodExact()
    x_eval, rho_ex, u_ex, p_ex = exact_solver.solve(t=0.2)
    
    # 预测
    x_test = torch.tensor(x_eval.reshape(-1, 1), dtype=torch.float32).to(device)
    t_test = torch.ones_like(x_test) * 0.2
    
    model.eval()
    with torch.no_grad():
        out = model(torch.cat([x_test, t_test], 1)).cpu().numpy()
    
    rho_pred, u_pred, p_pred = out[:, 0], out[:, 1], out[:, 2]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 密度
    axs[0].plot(x_eval, rho_ex, 'k-', lw=2, label='Exact')
    axs[0].plot(x_eval, rho_pred, 'r--', lw=2, label='R-PAG PINN')
    axs[0].set_title("Density (rho)")
    axs[0].legend()
    axs[0].grid()
    
    # 速度
    axs[1].plot(x_eval, u_ex, 'k-', lw=2, label='Exact')
    axs[1].plot(x_eval, u_pred, 'r--', lw=2, label='R-PAG PINN')
    axs[1].set_title("Velocity (u)")
    axs[1].grid()
    
    # 压力
    axs[2].plot(x_eval, p_ex, 'k-', lw=2, label='Exact')
    axs[2].plot(x_eval, p_pred, 'r--', lw=2, label='R-PAG PINN')
    axs[2].set_title("Pressure (p)")
    axs[2].grid()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()