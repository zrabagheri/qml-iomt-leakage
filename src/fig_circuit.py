import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
fig, ax = plt.subplots(figsize=(3.5, 1.55))
q = 4
for j in range(q):
    ax.plot([0.2, 7.6], [q - 1 - j] * 2, color='k', lw=0.7, zorder=0)
    ax.text(0.05, q - 1 - j, r'$|0\rangle$', ha='right', va='center')
def box(x, y, t, c):
    ax.add_patch(FancyBboxPatch((x - 0.32, y - 0.3), 0.64, 0.6, boxstyle='round,pad=0.02', fc=c, ec='k', lw=0.6, zorder=2))
    ax.text(x, y, t, ha='center', va='center', fontsize=6.5, zorder=3)
x0 = 0.7
for j in range(q):
    box(x0, q - 1 - j, r'$R_y(x_{%d})$' % (j + 1), '#dbe9f6')
    box(x0 + 0.85, q - 1 - j, r'$R_y(\theta_{l%d})$' % (j + 1), '#fde2c4')
xs = [2.45, 2.85, 3.25, 3.65]
for k, j in enumerate(range(q)):
    t = (j + 1) % q; x = xs[k]
    yc, yt = q - 1 - j, q - 1 - t
    ax.plot([x, x], [yc, yt], color='k', lw=0.7, zorder=1)
    ax.plot(x, yc, 'ko', ms=2.5, zorder=3)
    ax.add_patch(plt.Circle((x, yt), 0.13, fc='white', ec='k', lw=0.7, zorder=3))
    ax.plot([x - 0.13, x + 0.13], [yt, yt], 'k', lw=0.6, zorder=4); ax.plot([x, x], [yt - 0.13, yt + 0.13], 'k', lw=0.6, zorder=4)
ax.add_patch(plt.Rectangle((0.3, -0.5), 3.65, q, fill=False, ls='--', lw=0.6, ec='gray'))
ax.text(2.1, q - 0.35, r'layer $l$ (repeated $L=4$ times)', ha='center', fontsize=6.5, color='gray')
ax.text(5.3, 1.5, r'$\cdots$', fontsize=12, ha='center', va='center')
ax.add_patch(FancyBboxPatch((6.6, q - 1 - 0.3), 0.75, 0.6, boxstyle='round,pad=0.02', fc='#e3f1df', ec='k', lw=0.6))
ax.text(6.97, q - 1, r'$\langle Z_1\rangle$', ha='center', va='center', fontsize=7)
ax.text(6.97, q - 1.55, r'$\hat p=\sigma(w\langle Z_1\rangle+b)$', ha='center', va='center', fontsize=6.5, bbox=dict(fc='white', ec='none', pad=0.5), zorder=5)
ax.set_xlim(-0.4, 7.7); ax.set_ylim(-0.7, q - 0.1); ax.axis('off')
plt.tight_layout(pad=0.1)
plt.savefig('figs/Fig1_circuit.png', dpi=600)
