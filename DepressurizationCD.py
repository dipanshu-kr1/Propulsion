import numpy as np
import matplotlib.pyplot as plt
gamma = 1.4
d_e = 0.008
D_t = 0.008
A_e= 1.0079*np.pi * (D_t/2)**2
A_star = np.pi * (D_t/2)**2
T_0i = 300
p_0i = 20e5 #bar
p_b = 1e5 
C_d = 0.9
V = 0.0021 #m^3
R = 287.0   


K=(2/(gamma+1)**((gamma+1)/(2*(gamma-1))))
E=A_e/A_star
a_0i = np.sqrt(gamma * R * T_0i)

tau = V / (C_d * A_star * K * a_0i)
tau_e = V / (C_d * A_e * a_0i) 

def solve_mach(epsilon, branch="subsonic"):
    def f(M):
        term = (2.0 / (gamma + 1.0)) * (1.0 + ((gamma - 1.0) / 2.0) * M**2)
        exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return (1.0 / M) * (term ** exponent) - epsilon

    if branch == "subsonic":
        a, b = 1e-6, 1.0
    else:
        a, b = 1.0, 50.0

    for _ in range(100):
        m = (a + b) / 2.0
        if f(m) * f(a) > 0:
            a = m
        else:
            b = m
        if (b - a) < 1e-12:
            break
            
    return (a + b) / 2.0

M_sub = solve_mach(E, branch="subsonic")
M_sup = solve_mach(E, branch="supersonic")

def compute_pi(M):
    return (1.0 + ((gamma - 1.0) / 2.0) * M**2) ** (-gamma / (gamma - 1.0))

pi_sub = compute_pi(M_sub)
pi_des = compute_pi(M_sup)

shock_term = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (M_sup**2 - 1.0)
pi_ns = pi_des * shock_term

results = {}
regimes = {'des': pi_des, 'ns': pi_ns, 'sub': pi_sub}
    
for key, pi_val in regimes.items():
    p_k = p_b / pi_val
    
    # Line 9: t_k calculation with max(0, ...) safety check
    power_term = (p_k / p_0i) ** (- (gamma - 1.0) / (2.0 * gamma))
    t_k = max(0.0, (2.0 * tau / (gamma - 1.0)) * (power_term - 1.0))
        
    results[f'p_{key}'] = p_k
    results[f't_{key}'] = t_k

A_t = A_star  

def regime(p0):
    r = p_b / p0
    if r <= pi_des: return "A"
    if r < pi_ns:   return "B"
    if r < pi_sub:  return "C"
    return "D"

def mass_flow(p0, T0):
    r = p_b / p0
    if r < pi_sub:  # choked (A, B, C)
        return C_d * A_star * K * p0 * np.sqrt(gamma / (R * T0))
    r = min(r, 1.0)  # unchoked (D)
    return (C_d * A_e * p0 * np.sqrt(2*gamma / ((gamma-1) * R * T0))
            * r**(1/gamma) * np.sqrt(max(0.0, 1 - r**((gamma-1)/gamma))))

def T0_of_p(p0):
    return T_0i * (p0 / p_0i) ** ((gamma - 1) / gamma)

def dp0_dt(p0):
    T0 = T0_of_p(p0)
    return -gamma * R * T0 / V * mass_flow(p0, T0)

dt = 1e-3
t_sub, p_sub = results['t_sub'], results['p_sub']
hist = []

t, p0 = 0.0, p_0i
if p_0i > p_sub:
    while t < t_sub:
        theta = 1 + 0.5 * (gamma - 1) * t / tau
        p0 = p_0i * theta ** (-2*gamma / (gamma - 1))
        T0 = T_0i * theta ** (-2)
        hist.append((t, p0, T0, mass_flow(p0, T0), regime(p0)))
        t += dt
    t, p0 = t_sub, p_sub    

while p0 > p_b:
    k1 = dp0_dt(p0)
    k2 = dp0_dt(p0 + 0.5*dt*k1)
    k3 = dp0_dt(p0 + 0.5*dt*k2)
    k4 = dp0_dt(p0 + dt*k3)
    p0 = max(p_b, p0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4))
    t += dt
    T0 = T0_of_p(p0)
    hist.append((t, p0, T0, mass_flow(p0, T0), regime(p0)))

hist = np.array(hist, dtype=object)
t_arr  = hist[:, 0].astype(float)
p_arr  = hist[:, 1].astype(float)
T_arr  = hist[:, 2].astype(float)
m_arr  = hist[:, 3].astype(float)



eps_ratio = E  

def exit_state(p0, T0, reg):
    r = p_b / p0
    if reg in ("A", "B"):
        Me = M_sup
        pe = pi_des * p0
    elif reg == "C":
        Q = K / (eps_ratio * r)                                   
        Me = np.sqrt((-1 + np.sqrt(1 + 2*(gamma-1)*Q**2)) / (gamma-1))
        pe = p_b
    else:  
        Me = np.sqrt(max(0.0, 2/(gamma-1) * (r**(-(gamma-1)/gamma) - 1)))  
        pe = p_b
    Te = T0 / (1 + 0.5*(gamma-1)*Me**2)
    Ve = Me * np.sqrt(gamma * R * Te)
    return Me, pe, Ve

reg_arr = hist[:, 4]
F_arr  = np.zeros_like(t_arr)
Me_arr = np.zeros_like(t_arr)
for i in range(len(t_arr)):
    Me, pe, Ve = exit_state(p_arr[i], T_arr[i], reg_arr[i])
    F_arr[i]  = m_arr[i] * Ve + (pe - p_b) * A_e
    Me_arr[i] = Me



t_f = t_arr[-1]
bounds = [("A", 0.0,           results['t_des'], "tab:green"),
          ("B", results['t_des'], results['t_ns'],  "tab:orange"),
          ("C", results['t_ns'],  results['t_sub'], "tab:red"),
          ("D", results['t_sub'], t_f,              "tab:blue")]
bounds = [b for b in bounds if b[2] > b[1]]    # drop skipped regimes



def mark_regimes(ax, label=True):
    for name, t0, t1, col in bounds:
        ax.axvspan(t0, t1, color=col, alpha=0.12)
        if label:
            ax.text(0.5*(t0+t1), 0.97, name, color=col, ha="center", va="top",
                    fontweight="bold", transform=ax.get_xaxis_transform())
    for name, t0, t1, col in bounds[1:]:
        ax.axvline(t0, color="k", ls="--", lw=0.8)


fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

# Mark regime transition times on both plots
for k in ("des", "ns", "sub"):
    for a in ax:
        a.axvline(results[f"t_{k}"], color="k", ls="--", lw=0.8)

# Plot net thrust
ax[0].plot(t_arr, F_arr, "k")
ax[0].set_ylabel("Net thrust F [N]")
mark_regimes(ax[0])

# Plot exit Mach number
ax[1].plot(t_arr, Me_arr, "k")
ax[1].set_ylabel("Exit Mach number")
ax[1].set_xlabel("t [s]")
mark_regimes(ax[1], label=False)


for i, (name, t0, t1, col) in enumerate(bounds[1:], start=1):
    previous_name = bounds[i - 1][0]

    ax[0].annotate(
        f"{previous_name} → {name}\nt = {t0:.2f} s",
        xy=(t0, np.interp(t0, t_arr, F_arr)),
        xytext=(5, 20),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="-", lw=0.5)
    )

plt.tight_layout()
plt.show()


print(
    f"Peak thrust: {F_arr.max():.1f} N "
    f"at t = {t_arr[F_arr.argmax()]:.3f} s"
)

print(
    f"Total impulse: {np.trapezoid(F_arr, t_arr):.1f} N·s"
)

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 10))


ax[0].plot(t_arr, p_arr/1e5, color="tab:blue", lw=2, label=r"$p_0$ (tank)")
ax[0].axhline(p_b/1e5, color="gray", ls=":", lw=1.2, label=r"$p_b$ (back pressure)")
ax[0].set_ylabel(r"$p_0$ [bar]")
ax[0].set_title("Tank blowdown through a converging–diverging nozzle", fontweight="bold")

ax[1].plot(t_arr, T_arr, color="tab:red", lw=2, label=r"$T_0$ (tank)")
ax[1].axhline(T_0i, color="gray", ls=":", lw=1.2, label=r"$T_{0,i}$")
ax[1].set_ylabel(r"$T_0$ [K]")

ax[2].plot(t_arr, m_arr, color="tab:green", lw=2, label=r"$\dot m$")
ax[2].set_ylabel(r"$\dot m$ [kg/s]")
ax[2].set_xlabel("t [s]")


for i, a in enumerate(ax):
    mark_regimes(a, label=(i == 0))          # A/B/C/D letters on the top panel only
    a.grid(True, alpha=0.3)
    a.set_xlim(0, t_arr[-1])


curves = [p_arr/1e5, T_arr, m_arr]
fmts   = ["{:.2f} bar", "{:.1f} K", "{:.4f} kg/s"]
for k in range(1, len(bounds)):
    prev, cur = bounds[k-1][0], bounds[k][0]
    tt = bounds[k][1]
    for a, y, f in zip(ax, curves, fmts):
        yv = np.interp(tt, t_arr, y)
        a.plot(tt, yv, "ko", ms=5, zorder=5)
        a.annotate(f"{prev}→{cur}\nt={tt:.2f} s\n" + f.format(yv),
                   xy=(tt, yv), xytext=(8, 8), textcoords="offset points",
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
                   arrowprops=dict(arrowstyle="-", lw=0.5))


from matplotlib.patches import Patch
desc = {"A": "A: choked, ideal/under-expanded",
        "B": "B: choked, over-expanded",
        "C": "C: choked, shock in nozzle",
        "D": "D: unchoked, subsonic"}
handles = [Patch(color=col, alpha=0.3, label=desc[name]) for name, _, _, col in bounds]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)

for a in ax:
    a.legend(loc="center right", fontsize=8)

plt.tight_layout(rect=[0, 0.07, 1, 1])      # leave room for the regime legend
plt.show()