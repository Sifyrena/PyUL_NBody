import sys
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import solve_ivp

n = 2 #The number of fields
c = [1/3, 2/3] #the ratio of mass. e.g. c_i =  m_i/m_tot ------->m: mass of feild or axion
#dr = .00001
#max_radius = 9.0 # This may be too large for lower order states; for ground state up to 2 nodes 9 is enough, up to 5 nodes 18 is enough.
#rge = max_radius/dr
#r_span = np.linspace(dr/1000, rge * dr, int(rge)+1)

#Notice:
#df = [psi_1', psi_1'', beta_1',..........(repetitive but different fields), phi', phi''] ---------->there are 3n+2 terms ,single field parameters from [3i,3i+2] like---->0,1,2 / 3,4,5
#f = [psi_1, psi_1', beta,..........(repetitive but different fields), phi, phi']


def equations(r, f):       #--------->r should equal to dr*rge[i]--------->这里就是定义需要解的方程组
    df = np.zeros(3*n + 2)
    rho = 0.0 
    for i in range(n):
        ix = 3 * i 
        df[ix] = f[ix+1]
        df[ix+1] = -(2/r) * f[ix+1] + 2 * c[i] * (c[i] * f[3*n] + f[ix+2]) * f[ix]
        #df[ix+1] = -(2/r) * f[ix+1] + 2 * c[i] * (f[3*n] - f[ix+2]) * f[ix]
        df[ix+2] = 0
        rho += f[ix]**2 * c[i]
        #rho += f[ix]**2 
    df[3*n] = f[3*n+1]   
    df[3*n+1] = -(2/r) * f[3*n+1] + 4 * np.pi * rho
    #df[3*n+1] = -(2/r) * f[3*n+1] + rho
    return df


def Jacobian(r, f):    #------------->这里就是定义需要解的方程组的Jacobian matrix
    N = 3*n + 2                                                    #The number of equations and also means the number of unknow
    J = np.zeros((N, N))
    for i in range(n):
        ix  = 3 * i 
        #For ix terms 
        J[ix,ix+1] = 1 #Others equal 0.

        #For ix+1 terms
        J[ix+1, ix] = 2 * c[i] * (c[i] * f[3*n] + f[ix+2])       
        #J[ix+1, ix] = 2 * c[i] * (f[3*n] - f[ix+2])       
        J[ix+1, ix+1] = -2/r
        J[ix+1, ix+2] = 2 * c[i] * f[ix]
        #J[ix+1, ix+2] = -f[ix]
        J[ix+1, 3*n] = 2 * c[i] * c[i] * f[ix]
        #J[ix+1, 3*n] = f[ix]

        #For ix terms
        #they  are 0.

        #For 3n terms(only write the eqs relative to this loop.)
        #They have some terms not 0,but not in this loop

        #For 3n+1 terms(only write the eqs relative to this loop.)
        J[3*n+1, ix] = 4 * np.pi * 2 * f[ix] *c[i]
        #J[3*n+1, ix] = 2 * f[ix] 
    
    #Now for the 3n and 3n+1 terms they are not includ in the loop
    J[3*n, 3*n+1] = 1
    J[3*n+1, 3*n+1] = -2/r
    #0print(J)
    return J



def PDE_solver(f_ini, r_span, show_last_point):  #-------->这里是根据上面的方程组和jacobian来接方程得到对应的f,
                                                 #-------->所以我们需要initial value和定义域
                                                 #这里的f和上面的f不一样, 这里的f是全部空间的f,而上面的f是指某个r的f(就是某个点的f),但是对应的顺序还是一样的，只不过是包含全空间的点
                                                 #这一步就是
    
    
    sol = solve_ivp(fun = equations,
                    t_span = (r_span[0], r_span[-1]),
                    y0 = f_ini,
                    t_eval = r_span,
                    method = 'BDF',  # or 'BDF' if using jacobian
                    jac = Jacobian
                    )
    

    if show_last_point == True:
        f_last = sol.y[:, -1]
        return  f_last # return final state only（在最远地方的y）ps：f指很多方程的解不仅仅是psi的解------->这个matrix的列代表一个点的所有psi等等的数值，这里的-1代表当我们的点取最远时候的数值
    else:
        f = sol.y      # fill full trajectory into f(全空间的f)
        return  f
    

def Newton_Raphson(f_ini, beta, r_span):       #这里我们算是设置了一个新的函数关系就是f(r_max)和beta之间的关系应用------>newton raphson方法
    J = np.zeros((n,n))
    psi_end = np.zeros(n)
    dbeta = np.full(n, 1e-10)
    show_last_point = True                  #这里我们设置的True所以这个function内的PDE_solver都是解关于最后一个点的信息

    f_last = PDE_solver(f_ini, r_span, show_last_point)

    for i in range(n):
        psi_end[i] = f_last[3*i]


    # Build Jacobian via finite difference
    for j in range(n):
        f_ini_perturbed = f_ini.copy()
        f_ini_perturbed[3*j+2] = beta[j] + dbeta[j]  # Perturb beta[j]
        
        f_dum = PDE_solver(f_ini_perturbed, r_span, show_last_point)    #Here f_dum means dummy variable  # Final f for perturbed beta
                                                                        #在每一个beta（第一个第二个场）变化之后f数列的变化

        for i in range(n):
            J[i, j] = (f_dum[3 * i] - psi_end[i]) / dbeta[j]            #因为上面一个comment的原因，所以每一个场与两个场的beta值变化都有关系
                                                                        #这里就是perturb一点beta是为了求这一点beta的导数对吗（也就是jacobain matrix）
                                                                        #然后再求在这个斜率下，psi_end到0，beta的变化数值dbeta
    dbeta = np.linalg.solve(J, -psi_end)                                #这一步类似于，x_n+1 - xn = -J*F
    # Update beta and f_ini
    beta += dbeta                                                 #这一步beta更新了就是在这个斜率下的切线为0时候的beta值，下面用这个接着去代入求解
    for k in range(n):
        f_ini[3*k+2] = beta[k]  # Update the last Na elements of y0 #这一步就是调整f_ini的数值然后在代入PDE_solver中再次求解，再看f[:,-1] #这里的f_ini中的psi还是上一个的beta对应的值
    return f_ini


def shooting_progress_test(f_ini, r_span, grid, dr):
    # 初始设置

    show_last_point = False

    #grid = 400
    #dr = 0.01 

    # 初始 beta
    beta = np.zeros(n)
    for i in range(n):
        beta[i] = f_ini[3*i + 2]  # 初始估计 β 来自 f 的末尾
    print("Initial β:", beta)

    kend = 20
    for k in range(kend):
        for i in range(50):
            f_ini = Newton_Raphson(f_ini, beta, r_span)
            for j in range(n):
                beta[j] = f_ini[3*j + 2] 
            print(f"[k={k}, i={i}] Beta =", beta)
            f = PDE_solver(f_ini, r_span, show_last_point)
        if k < kend - 1:
            grid += 50
            r_span = np.linspace(dr, grid * dr, grid)
            f = np.zeros((3*n+2, grid))
            f = PDE_solver(f_ini, r_span, show_last_point)

    for i in range(n):
        psi_index = 3 * i  # psi_i 在 f 中的位置
        f[psi_index, f[psi_index, :] < 0] = 0  
    
    print('the number of grid = ', grid)
    return f, r_span


    


def shooting_progress(f_ini, r_span, grid, dr):
    # 初始设置
    converged = False
    max_iter = 1000  # 防止无限循环
    iter_count = 0
    show_last_point = False

    #grid = 400
    #dr = 0.01 

    # 初始 beta
    beta = np.zeros(n)
    for i in range(n):
        beta[i] = f_ini[3*i + 2]  # 初始估计 β 来自 f 的末尾
    print("Initial β:", beta)

    # 主循环
    while not converged and iter_count < max_iter:
        iter_count += 1
        beta_prev = beta.copy()

    # 用 Newton-Raphson 修正 beta 和 f_ini
        f_ini = Newton_Raphson(f_ini, beta, r_span)
        f = PDE_solver(f_ini, r_span, show_last_point)

    # 计算变化量和尾部误差
        for i in range(n):
            beta[i] = f_ini[3*i + 2]
        dbeta = beta - beta_prev
        psi_end = np.array([f[3*i, -1] for i in range(n)])

        print(f"Iteration {iter_count}: β = {beta}, ψ_end = {psi_end}")

    # 收敛条件：beta 改变量 和 ψ_end 都足够小
        if np.max(dbeta) < 1e-7 and np.max(np.abs(psi_end)) < 1e-7:
            print(f"Converged at iteration {iter_count}")
            converged = True
            break

    # 每 50 次扩展一次 r 区间
        if iter_count % 50 == 0:
            rge += 50
            r_span = np.linspace(dr / 1000, (rge - 1) * dr, int(rge))
            f = PDE_solver(f_ini, r_span, show_last_point)

    return f, r_span



grid = 700
dr = 0.01
r_span = np.linspace(dr / 1000, (grid - 1) * dr, int(grid))

psi_ini = [1.0, 2.0]           #central ratio of wavefunction
beta_ini = [-3.1, -4.1]             #inital guess of beta
    
#For collection f
f_ini = np.array([psi_ini[0], 0.0, beta_ini[0], psi_ini[1], 0.0, beta_ini[1], 0.0, 0.0]) #---------->there are 3n+2 terms ,single field parameters from [3i,3i+2] like---->0,1,2 / 3,4,5


f, r_span = shooting_progress_test(f_ini, r_span, grid, dr)

# Plot ψ₁ and ψ₂
plt.figure(figsize=(8, 5))
for i in range(n):
    psi_index = 3 * i
    plt.plot(r_span, f[psi_index, :], label=fr'$\psi_{{{i+1}}}(r)$')
plt.xlabel("r", fontsize=14)
plt.ylabel("Wavefunction amplitude", fontsize=14)
plt.title("Two-field Soliton Profiles", fontsize=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()