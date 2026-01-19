import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from collections import OrderedDict
import time

if not os.path.isdir('./models'):
    os.mkdir('./models')

# Domain parameters
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
U_lid = 1.0
nu = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

same_seed(42)

# ==========================================
# 1. Neural Network
# ==========================================
class PINN(nn.Module):  
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        depth,
        act=torch.nn.Tanh,
    ):
        super(PINN, self).__init__()
        
        layers = [('input', torch.nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))

        for i in range(depth): 
            layers.append(('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size)))
            layers.append(('activation_%d' % i, act()))

        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))

        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)

        # 权重初始化
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        out = self.layers(x)
        u, v, p = out[:, 0], out[:, 1], out[:, 2]  
        return u, v, p

# ==========================================
# 2. R-PAG V5.0 Solver
# ==========================================
class LDC_RPAGSolver:
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

    def compute_residuals_vector(self, X_f):
        """计算逐点的 PDE 残差向量"""
        X_f.requires_grad_(True)
        u, v, p = self.model(X_f)
        
        # First derivatives
        u_grad = torch.autograd.grad(u, X_f, torch.ones_like(u), create_graph=True)[0]
        u_x, u_y = u_grad[:, 0], u_grad[:, 1]
        
        v_grad = torch.autograd.grad(v, X_f, torch.ones_like(v), create_graph=True)[0]
        v_x, v_y = v_grad[:, 0], v_grad[:, 1]
        
        p_grad = torch.autograd.grad(p, X_f, torch.ones_like(p), create_graph=True)[0]
        p_x, p_y = p_grad[:, 0], p_grad[:, 1]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, X_f, torch.ones_like(u_x), create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y, X_f, torch.ones_like(u_y), create_graph=True)[0][:, 1]
        
        v_xx = torch.autograd.grad(v_x, X_f, torch.ones_like(v_x), create_graph=True)[0][:, 0]
        v_yy = torch.autograd.grad(v_y, X_f, torch.ones_like(v_y), create_graph=True)[0][:, 1]
        
        # Navier-Stokes Equations
        continuity = u_x + v_y
        momentum_u = u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) 
        momentum_v = u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
        
        # 返回每个点的残差平方和
        res_sq = continuity**2 + momentum_u**2 + momentum_v**2
        return res_sq

    def step(self, X_f, X_b, u_b, v_b, use_rpag=True):
        self.optimizer.zero_grad()
        
        # 1. 计算内部 PDE 残差 (Vector)
        res_sq = self.compute_residuals_vector(X_f)
        
        # 2. 计算边界 BC Loss (Scalar)
        u_pred, v_pred, _ = self.model(X_b)
        bc_loss = torch.mean((u_pred - u_b)**2) + torch.mean((v_pred - v_b)**2)
        
        is_proj = False

        # === Warm-up 阶段 ===
        if not use_rpag:
            pde_loss = torch.mean(res_sq)
            loss = pde_loss + bc_loss * 10.0 
            loss.backward()
            self.optimizer.step()
            return loss.item(), False

        # === R-PAG 阶段 ===
        else:
            # 1. 动态分组
            res_abs = torch.sqrt(res_sq).detach()
            threshold = torch.quantile(res_abs, 0.90) # Top 10%
            
            mask_high = (res_abs >= threshold).float()
            mask_low = 1.0 - mask_high
            
            L_high = torch.sum(res_sq * mask_high) / (torch.sum(mask_high) + 1e-8)
            L_low = torch.sum(res_sq * mask_low) / (torch.sum(mask_low) + 1e-8)
            
            # 2. 梯度计算 
            g_high_pde = self.get_gradients(L_high * 2.0)
            g_bc = self.get_gradients(bc_loss * 10.0) 
            
            g_high = {}
            for name in g_high_pde:
                g_high[name] = g_high_pde[name] + g_bc[name]
                
            g_low = self.get_gradients(L_low * 1.0)
            
            # 3. GCond
            g_low_stable = {}
            for name in g_low:
                self.V_smooth[name] = self.beta * self.V_smooth[name] + (1 - self.beta) * g_low[name]
                g_low_stable[name] = self.V_smooth[name]
            
            # 4. 非对称投影
            g_low_final, is_proj = self.asymmetric_projection(g_low_stable, g_high)
            
            # 5. 更新参数
            for name, param in self.model.named_parameters():
                param.grad = g_high[name] + g_low_final[name]
                
            self.optimizer.step()
            
            total_loss = L_high + L_low + bc_loss
            return total_loss.item(), is_proj

# ==========================================
# 3. Data Generation
# ==========================================
N_f = 10000
N_b = 2000

# Collocation points
x_f = torch.rand((N_f, 1), dtype=torch.float32) * (x_max - x_min) + x_min
y_f = torch.rand((N_f, 1), dtype=torch.float32) * (y_max - y_min) + y_min
X_f = torch.cat([x_f, y_f], dim=1).to(device)

# Boundary points generation
def generate_boundary():
    n = N_b // 4
    # Left Wall (x = 0)
    left = torch.cat([x_min * torch.ones((n, 1), device=device), 
                      torch.rand((n, 1), device=device) * (y_max - y_min) + y_min], dim=1)
    # Right Wall (x = 1)
    right = torch.cat([x_max * torch.ones((n, 1), device=device), 
                       torch.rand((n, 1), device=device) * (y_max - y_min) + y_min], dim=1)
    # Bottom Wall (y = 0)
    bottom = torch.cat([torch.rand((n, 1), device=device) * (x_max - x_min) + x_min, 
                        y_min * torch.ones((n, 1), device=device)], dim=1)
    # Top Lid (y = 1)
    top = torch.cat([torch.rand((n, 1), device=device) * (x_max - x_min) + x_min, 
                     y_max * torch.ones((n, 1), device=device)], dim=1)
    return torch.cat([left, right, bottom, top], dim=0)

X_b = generate_boundary()

# Boundary conditions
u_top = U_lid * torch.ones((N_b // 4,), device=device) 
v_top = torch.zeros((N_b // 4,), device=device)
u_side = torch.zeros((3 * N_b // 4,), device=device)
v_side = torch.zeros((3 * N_b // 4,), device=device)

u_b = torch.cat([u_side, u_top], dim=0) 
v_b = torch.cat([v_side, v_top], dim=0)

# ==========================================
# 4. Main Training Loop
# ==========================================
model = PINN(
        input_size=2,
        hidden_size=20,
        output_size=3,
        depth=8,
        act=torch.nn.Tanh
    ).to(device)

# 使用 R-PAG Solver
solver = LDC_RPAGSolver(model, lr=1e-3)

best_loss = math.inf
epochs = 30000

print(f">>> Start Training with R-PAG V5.0 (Warm-up included) <<<")
start_time = time.time()

for epoch in range(epochs):
    # Warm-up 策略: 前 20% 轮次不开启 R-PAG
    if epoch < epochs * 0.2:
        use_rpag = False
    else:
        use_rpag = True
        
    loss_val, is_proj = solver.step(X_f, X_b, u_b, v_b, use_rpag=use_rpag)

    # 保存最佳模型
    if loss_val < best_loss:
        best_loss = loss_val
        torch.save(model.state_dict(), './models/model.ckpt')
    
    if epoch % 500 == 0:
        mode = "R-PAG" if use_rpag else "WarmUp"
        proj_str = "YES" if is_proj else "NO"
        print(f"Epoch {epoch:05d} | Mode: {mode} | Loss: {loss_val:.5f} | Proj: {proj_str}")

print(f"Training finished in {time.time()-start_time:.2f}s")
print(f"Best Loss: {best_loss:.5f}")

# ==========================================
# 5. Visualization (修改为复现速度大小分布图)
# ==========================================
# 增加分辨率以获得平滑的云图效果
nx, ny = 200, 200 
x = torch.linspace(x_min, x_max, nx, device=device)
y = torch.linspace(y_min, y_max, ny, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')
XY = torch.stack([X.ravel(), Y.ravel()], -1)

# 加载模型
model.load_state_dict(torch.load('models/model.ckpt', map_location=device, weights_only=True))
model.eval()

with torch.no_grad():
    u, v, p = model(XY)
    # 转换为 numpy 数组
    u = u.reshape(ny, nx).cpu().numpy()
    v = v.reshape(ny, nx).cpu().numpy()

# 计算合速度大小
speed = np.sqrt(u**2 + v**2)

# 绘制速度大小分布图 (复现目标图片风格)
plt.figure(figsize=(8, 6), dpi=100)
# 使用 imshow 绘制热力图，cmap='jet' 对应蓝-红配色，origin='lower' 对应原点在左下角
plt.imshow(speed, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='jet', interpolation='nearest')

cbar = plt.colorbar()
cbar.set_label('Speed Magnitude')

plt.title('Velocity Magnitude Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()