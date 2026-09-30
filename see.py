import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
# ===== PARAMETERS ( do all the changes here )
GRID_SIZE = 20
EMPTY_FRAC = 0.2
A_FRAC = 0.4
B_FRAC = 0.4
THRESHOLD = 0.3
NUM_STEPS = 100
SNAPSHOT_AT = [0, 25, 50, 100]


# ===== EXPERIMENT B PARAMETERS =====
T_TOLERANT   = 0.2   # tolerant agents — happy with just 20% same-type neighbours
T_INTOLERANT = 0.7   # intolerant agents — need 70% same-type neighbours
X_INTOLERANT = 0.2   # test with 20% intolerant agents first
GRID_B       = 50    # grid size for experiment B

#B3 parameters 
X_VALUES  = [i/100 for i in range(0, 101, 10)]  # 0.0, 0.1 ... 1.0
TRIALS_B  = 15      # 5 runs per X value to average out randomness
STEPS_B   = 150    # steps to reach convergence

# ===== EXPERIMENT C PARAMETERS =====
T_UNIFORM  = 0.4    # everyone has this threshold (uniform population)
T_LOW      = 0.1    # tolerant half in bimodal
T_HIGH     = 0.7    # intolerant half in bimodal
GRID_C     = 50     # same grid size as B
TRIALS_C   = 15     # same trial count as B
STEPS_C    = 150    # same steps as B
# ===== BUILD THE GRID 
probs = [EMPTY_FRAC, A_FRAC, B_FRAC]
grid = np.random.choice([0, 1, 2], size=(GRID_SIZE, GRID_SIZE), p=probs)

# ===== TASK 1: DISPLAY INITIAL GRID 
cmap = mcolors.ListedColormap(['white', 'blue', 'red'])
plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap=cmap, vmin=0, vmax=2)
plt.title("Initial Grid")
plt.axis('off')
plt.show()
 
print("Grid shape       :", grid.shape)
print("Empty cells      :", np.sum(grid == 0))
print("Group A cells    :", np.sum(grid == 1))
print("Group B cells    :", np.sum(grid == 2))


# ===== FUNCTIONS 

def get_neighbors(grid, row, col):
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r = row + dr
            c = col + dc
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                if grid[r][c] != 0:
                    neighbors.append(grid[r][c])
    return neighbors


def is_happy(grid, row, col, threshold):
    agent = grid[row][col]
    if agent == 0:
        return None
    neighbors = get_neighbors(grid, row, col)
    if len(neighbors) == 0:
        return True
    same = sum(1 for n in neighbors if n == agent)
    ratio = same / len(neighbors)
    return ratio >= threshold


def run_one_step(grid, threshold):
    unhappy = []
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                if not is_happy(grid, row, col, threshold):
                    unhappy.append((row, col))
    empty = []
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 0:
                empty.append((row, col))
    random.shuffle(unhappy)
    for (r, c) in unhappy:
        if len(empty) == 0:
            break
        idx = random.randint(0, len(empty) - 1)
        new_r, new_c = empty[idx]
        grid[new_r][new_c] = grid[r][c]
        grid[r][c] = 0
        empty[idx] = (r, c)
    return grid


def get_happiness_percent(grid, threshold):    # ← fix 3: moved up here with all functions
    happy = 0
    total = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                total += 1
                if is_happy(grid, row, col, threshold):
                    happy += 1
    if total == 0:
        return 0
    return round(happy / total * 100, 1)
    
def create_threshold_grid(grid, x_intolerant, t_tolerant, t_intolerant):
    """Returns a threshold grid — same shape as agent grid.
    Each agent cell gets either t_tolerant or t_intolerant.
    x_intolerant = fraction of agents who are intolerant (0.0 to 1.0)
    """
    # start: everyone is tolerant
    tgrid = np.full(grid.shape, t_tolerant, dtype=float)

    # find all agent positions — non-empty cells only
    agent_positions = np.argwhere(grid != 0)

    # how many agents should be intolerant?
    n_intolerant = int(len(agent_positions) * x_intolerant)

    # pick that many positions randomly, no repetition
    chosen_indices = np.random.choice(
        len(agent_positions), n_intolerant, replace=False
    )

    # assign intolerant threshold to chosen agents
    for idx in chosen_indices:
        r, c = agent_positions[idx]
        tgrid[r][c] = t_intolerant

    return tgrid

def is_happy_hetero(grid, threshold_grid, row, col):
    """
    Same as is_happy — but each agent uses their own personal threshold
    from threshold_grid[row][col] instead of one global value.
    """
    agent = grid[row][col]
    if agent == 0:
        return None
    threshold = threshold_grid[row][col]   # ← personal threshold
    neighbors = get_neighbors(grid, row, col)
    if len(neighbors) == 0:
        return True
    same  = sum(1 for n in neighbors if n == agent)
    ratio = same / len(neighbors)
    return ratio >= threshold

def run_one_step_hetero(grid, threshold_grid):
    """
    Same as run_one_step — but uses is_happy_hetero.
    IMPORTANT: threshold travels with the agent when it moves.
    """
    # find unhappy agents using personal thresholds
    unhappy = []
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                if not is_happy_hetero(grid, threshold_grid, row, col):
                    unhappy.append((row, col))

    # find empty cells
    empty = []
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 0:
                empty.append((row, col))

    random.shuffle(unhappy)

    for (r, c) in unhappy:
        if len(empty) == 0:
            break
        idx = random.randint(0, len(empty) - 1)
        new_r, new_c = empty[idx]

        # move agent
        grid[new_r][new_c] = grid[r][c]
        grid[r][c] = 0

        # threshold travels with agent ← this is the key new line
        threshold_grid[new_r][new_c] = threshold_grid[r][c]
        threshold_grid[r][c] = T_TOLERANT  # vacated cell resets

        empty[idx] = (r, c)

    return grid, threshold_grid

def create_bimodal_threshold_grid(grid, t_low, t_high):
    """
    Bimodal distribution — exactly 50% of agents get t_low,
    exactly 50% get t_high.
    Mean threshold = (t_low + t_high) / 2
    """
    tgrid = np.full(grid.shape, t_low, dtype=float)

    agent_positions = np.argwhere(grid != 0)
    n_agents = len(agent_positions)

    # exactly half get t_high
    n_high = n_agents // 2
    chosen = np.random.choice(n_agents, n_high, replace=False)

    for idx in chosen:
        r, c = agent_positions[idx]
        tgrid[r][c] = t_high

    return tgrid


#DI value calculation
def get_dissimilarity_index(grid, patch_size=4):
    """
    Dissimilarity Index — standard census measure of segregation.
    
    Formula: D = (1/2) × Σ | a_i/A - b_i/B |
    
    Grid is divided into non-overlapping square patches (like census tracts).
    Each patch counts its Group A and Group B agents.
    
    Returns: 0.0 (fully integrated) to 1.0 (fully segregated)
    """
    total_A = np.sum(grid == 1)
    total_B = np.sum(grid == 2)

    # Safety check — if one group has vanished entirely
    if total_A == 0 or total_B == 0:
        return 1.0

    grid_size = len(grid)
    di = 0.0

    # Step through the grid in patch_size steps — non-overlapping patches
    for row_start in range(0, grid_size, patch_size):
        for col_start in range(0, grid_size, patch_size):

            # Extract this patch (numpy slicing)
            patch = grid[row_start : row_start + patch_size,
                        col_start : col_start + patch_size]

            a_i = np.sum(patch == 1)   # Group A count in this patch
            b_i = np.sum(patch == 2)   # Group B count in this patch

            # This patch's contribution to DI
            di += abs(a_i / total_A - b_i / total_B)

    return round(di / 2, 4)


# ===== TASK 2: COUNT HAPPY VS UNHAPPY =====
happy_count = 0
unhappy_count = 0

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        if grid[row][col] != 0:
            if is_happy(grid, row, col, THRESHOLD):
                happy_count += 1
            else:
                unhappy_count += 1

total = happy_count + unhappy_count
print("Happy agents   :", happy_count)
print("Unhappy agents :", unhappy_count)
print("Happiness %    :", round(happy_count / total * 100, 1), "%")

# ===== TASK 2: HAPPINESS MAP =====
happiness_grid = np.zeros((GRID_SIZE, GRID_SIZE))
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        if grid[row][col] != 0:
            if is_happy(grid, row, col, THRESHOLD):
                happiness_grid[row][col] = 1
            else:
                happiness_grid[row][col] = 2

cmap2 = mcolors.ListedColormap(['white', 'green', 'orange'])
plt.figure(figsize=(6, 6))
plt.imshow(happiness_grid, cmap=cmap2, vmin=0, vmax=2)
plt.title(f"Happiness Map (threshold = {THRESHOLD})")
plt.axis('off')
plt.show()

# introducing toroidal (wrap-around) neighborhood functions for comparison
def get_neighbors_toroidal(grid, row, col):
    """
    Toroidal (wrap-around) neighbourhood.
    Left edge connects to right. Top connects to bottom.
    Every single agent always has exactly 8 neighbours — no edge bias.
    Only change from get_neighbors: the two % lines that wrap coordinates.
    """
    neighbors = []
    grid_rows = len(grid)
    grid_cols = len(grid[0])

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r = (row + dr) % grid_rows   # wraps: row -1 becomes last row
            c = (col + dc) % grid_cols   # wraps: col -1 becomes last col
            if grid[r][c] != 0:
                neighbors.append(grid[r][c])
    return neighbors


def is_happy_toroidal(grid, row, col, threshold):
    """
    Same logic as is_happy — but uses toroidal neighbours.
    """
    agent = grid[row][col]
    if agent == 0:
        return None
    neighbors = get_neighbors_toroidal(grid, row, col)
    if len(neighbors) == 0:
        return True
    same  = sum(1 for n in neighbors if n == agent)
    ratio = same / len(neighbors)
    return ratio >= threshold


def run_one_step_toroidal(grid, threshold):
    """
    Same as run_one_step — but unhappy check uses toroidal neighbours.
    """
    unhappy = []
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                if not is_happy_toroidal(grid, row, col, threshold):
                    unhappy.append((row, col))
    empty = []
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 0:
                empty.append((row, col))
    random.shuffle(unhappy)
    for (r, c) in unhappy:
        if len(empty) == 0:
            break
        idx = random.randint(0, len(empty) - 1)
        new_r, new_c = empty[idx]
        grid[new_r][new_c] = grid[r][c]
        grid[r][c] = 0
        empty[idx] = (r, c)
    return grid

def get_happiness_percent_toroidal(grid, threshold):
    """Happiness check using toroidal neighbours."""
    happy = 0
    total = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                total += 1
                if is_happy_toroidal(grid, row, col, threshold):
                    happy += 1
    if total == 0:
        return 0
    return round(happy / total * 100, 1)

# ===== TASK 3: ONE STEP BEFORE/AFTER 
grid_before = grid.copy()
run_one_step(grid, THRESHOLD)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(grid_before, cmap=cmap, vmin=0, vmax=2)
ax1.set_title("Before (Step 0)")
ax1.axis('off')
ax2.imshow(grid, cmap=cmap, vmin=0, vmax=2)
ax2.set_title("After (Step 1)")
ax2.axis('off')
plt.tight_layout()
plt.show()

# ===== TASK 4: FULL SIMULATION (with DI in titles)
snapshots = {}
snapshots[0] = grid.copy()

for step in range(1, NUM_STEPS + 1):
    run_one_step(grid, THRESHOLD)
    if step in SNAPSHOT_AT:
        snapshots[step] = grid.copy()

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for i, step in enumerate(SNAPSHOT_AT):
    di_val   = get_dissimilarity_index(snapshots[step], patch_size=4)
    happy_val = get_happiness_percent(snapshots[step], THRESHOLD)
    axes[i].imshow(snapshots[step], cmap=cmap, vmin=0, vmax=2)
    axes[i].set_title(
        f"Step {step}\nDI = {di_val:.3f}   |   Happy = {happy_val}%",
        fontsize=11
    )
    axes[i].axis('off')

plt.suptitle(
    f"Schelling Segregation — T={THRESHOLD}  |  Grid={GRID_SIZE}×{GRID_SIZE}",
    fontsize=13
)
plt.tight_layout()
plt.show()

# ===== DI VERIFICATION: print DI at each snapshot =====
print("\n--- Dissimilarity Index at each step ---")
for step in SNAPSHOT_AT:
    di_val = get_dissimilarity_index(snapshots[step], patch_size=4)
    happy_val = get_happiness_percent(snapshots[step], THRESHOLD)
    print(f"Step {step:>3}  →  DI = {di_val:.4f}   |   Happiness = {happy_val}%")


# ===== TASK 5: TRACK HAPPINESS OVER TIME 
grid2 = np.random.choice([0, 1, 2], size=(GRID_SIZE, GRID_SIZE), p=probs)
happiness_over_time = []
happiness_over_time.append(get_happiness_percent(grid2, THRESHOLD))  # step 0

for step in range(1, NUM_STEPS + 1):
    run_one_step(grid2, THRESHOLD)
    pct = get_happiness_percent(grid2, THRESHOLD)   
    happiness_over_time.append(pct)

plt.figure(figsize=(8, 5))
plt.plot(happiness_over_time, color='green', linewidth=2)
plt.xlabel("Step")
plt.ylabel("Happy agents (%)")
plt.title(f"Happiness over time (threshold = {THRESHOLD})")
plt.ylim(0, 100)
plt.grid(True)
plt.show()

# ===== TASK 6: DI AND HAPPINESS TRACKED TOGETHER OVER TIME
grid3 = np.random.choice([0, 1, 2], size=(GRID_SIZE, GRID_SIZE), p=probs)

di_over_time       = []
happiness_over_time2 = []

# Step 0 — before any movement
di_over_time.append(get_dissimilarity_index(grid3, patch_size=4))
happiness_over_time2.append(get_happiness_percent(grid3, THRESHOLD))

# Steps 1 to NUM_STEPS
for step in range(1, NUM_STEPS + 1):
    run_one_step(grid3, THRESHOLD)
    di_over_time.append(get_dissimilarity_index(grid3, patch_size=4))
    happiness_over_time2.append(get_happiness_percent(grid3, THRESHOLD))

# Plot both on the same figure — two panels
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(happiness_over_time2, color='green', linewidth=2.5, label='Happiness %')
ax1.set_ylabel("Happy Agents (%)", fontsize=12)
ax1.set_ylim(0, 105)
ax1.axhline(y=100, color='green', linestyle='--', alpha=0.3)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.4)
ax1.set_title(
    f"Happiness and Dissimilarity Index over time  (T={THRESHOLD})",
    fontsize=13
)

ax2.plot(di_over_time, color='crimson', linewidth=2.5, label='Dissimilarity Index (DI)')
ax2.set_ylabel("DI  (0=integrated, 1=segregated)", fontsize=12)
ax2.set_xlabel("Simulation Step", fontsize=12)
ax2.set_ylim(0, 1.0)
ax2.axhline(y=0.6, color='gray', linestyle='--', alpha=0.4, label='DI = 0.6 reference')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

# Print final values clearly
print(f"\nFinal DI        : {di_over_time[-1]:.4f}")
print(f"Final Happiness : {happiness_over_time2[-1]}%")
print(f"DI at step 0    : {di_over_time[0]:.4f}")
print(f"DI increase     : +{di_over_time[-1] - di_over_time[0]:.4f}")

# ===== TASK 7: PATCH HEATMAP — WHERE IS SEGREGATION HAPPENING?

def show_patch_heatmap(grid, patch_size=4, title="Patch Segregation Map"):
    """
    Divides grid into patches and colors each patch by its Group A fraction.
    Red  = patch is mostly Group A (pure)
    Blue = patch is mostly Group B (pure)  
    White/grey = patch is mixed (integrated)
    """
    grid_size  = len(grid)
    n_patches  = grid_size // patch_size
    patch_map  = np.full((n_patches, n_patches), 0.5)  # 0.5 = perfectly mixed

    for i, row_start in enumerate(range(0, grid_size, patch_size)):
        for j, col_start in enumerate(range(0, grid_size, patch_size)):
            patch   = grid[row_start : row_start + patch_size,
                        col_start : col_start + patch_size]
            a_count = np.sum(patch == 1)
            b_count = np.sum(patch == 2)
            total   = a_count + b_count
            if total > 0:
                patch_map[i][j] = a_count / total  # 1.0=pure A, 0.0=pure B
            else:
                patch_map[i][j] = 0.5              # empty patch = neutral

    plt.figure(figsize=(5, 5))
    im = plt.imshow(patch_map, cmap='RdBu', vmin=0, vmax=1)
    plt.colorbar(im, label='← Pure B     Mixed     Pure A →')
    plt.title(title, fontsize=11)
    plt.xticks(range(n_patches), [f"C{j+1}" for j in range(n_patches)], fontsize=7)
    plt.yticks(range(n_patches), [f"R{i+1}" for i in range(n_patches)], fontsize=7)
    plt.tight_layout()
    plt.show()

# Show patch heatmap at step 0 (random) and step 100 (segregated)
show_patch_heatmap(
    snapshots[0],
    patch_size=4,
    title=f"Patch Map at Step 0  |  DI = {get_dissimilarity_index(snapshots[0]):.3f}"
)
show_patch_heatmap(
    snapshots[100],
    patch_size=4,
    title=f"Patch Map at Step 100  |  DI = {get_dissimilarity_index(snapshots[100]):.3f}"
)


# ===== EXPERIMENT A: GRID SIZE COMPARISON =====
    # Question: Does city size affect how segregated it becomes?
    # T=0.3 fixed. patch_size FIXED at 4 across all grids — same ruler always.

print("\n" + "="*60)
print("EXPERIMENT A — Grid Size vs Segregation (DI)")
print("="*60)

GRID_SIZES = [20, 50, 100]
TRIALS_A   = 3
STEPS_A    = 200
T_A        = 0.3

size_results = {}

for gs in GRID_SIZES:

    patch_s = 4           # ← FIXED at 4 for all grid sizes — same ruler
    di_list = []
    hp_list = []

    print(f"\n  Grid {gs}×{gs}  |  patch=4×4 (fixed)  |  {TRIALS_A} trials...")

    for trial in range(TRIALS_A):

        # ── Normal grid ──────────────────────────────────────
        g_normal = np.random.choice(
            [0, 1, 2], size=(gs, gs), p=[EMPTY_FRAC, A_FRAC, B_FRAC]
        )
        g_normal_initial = g_normal.copy()   # save BEFORE running

        for _ in range(STEPS_A):
            run_one_step(g_normal, T_A)

        di_n = get_dissimilarity_index(g_normal, patch_size=patch_s)
        hp_n = get_happiness_percent(g_normal, T_A)

        # ── Toroidal grid ─────────────────────────────────────
        g_torus = np.random.choice(
            [0, 1, 2], size=(gs, gs), p=[EMPTY_FRAC, A_FRAC, B_FRAC]
        )
        g_torus_initial = g_torus.copy()     # save BEFORE running

        for _ in range(STEPS_A):
            run_one_step_toroidal(g_torus, T_A)

        di_t = get_dissimilarity_index(g_torus, patch_size=patch_s)
        hp_t = get_happiness_percent_toroidal(g_torus, T_A)

        di_list.append((di_n, di_t))
        hp_list.append((hp_n, hp_t))

        print(f"    Trial {trial+1}: "
            f"Normal DI={di_n:.3f} ({hp_n}%)  |  "
            f"Toroidal DI={di_t:.3f} ({hp_t}%)")

        # ── Visual output — Trial 1 only, one figure per grid size ──
        if trial == 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Top-left: Normal grid at step 0
            axes[0][0].imshow(g_normal_initial, cmap=cmap, vmin=0, vmax=2)
            axes[0][0].set_title(
                f"Normal {gs}×{gs} — Step 0\n"
                f"DI = {get_dissimilarity_index(g_normal_initial, patch_size=4):.3f}",
                fontsize=11
            )
            axes[0][0].axis('off')

            # Top-right: Normal grid after STEPS_A steps
            axes[0][1].imshow(g_normal, cmap=cmap, vmin=0, vmax=2)
            axes[0][1].set_title(
                f"Normal {gs}×{gs} — Step {STEPS_A}\n"
                f"DI = {di_n:.3f}  |  Happy = {hp_n}%",
                fontsize=11
            )
            axes[0][1].axis('off')

            # Bottom-left: Toroidal grid at step 0
            axes[1][0].imshow(g_torus_initial, cmap=cmap, vmin=0, vmax=2)
            axes[1][0].set_title(
                f"Toroidal {gs}×{gs} — Step 0\n"
                f"DI = {get_dissimilarity_index(g_torus_initial, patch_size=4):.3f}",
                fontsize=11
            )
            axes[1][0].axis('off')

            # Bottom-right: Toroidal grid after STEPS_A steps
            axes[1][1].imshow(g_torus, cmap=cmap, vmin=0, vmax=2)
            axes[1][1].set_title(
                f"Toroidal {gs}×{gs} — Step {STEPS_A}\n"
                f"DI = {di_t:.3f}  |  Happy = {hp_t}%",
                fontsize=11
            )
            axes[1][1].axis('off')

            plt.suptitle(
                f"Grid {gs}×{gs} — Normal vs Toroidal  (T={T_A}, patch=4×4)",
                fontsize=13, fontweight='bold'
            )
            plt.tight_layout()
            plt.show()

    # Store averaged results for this grid size
    size_results[gs] = {
        'avg_di_normal' : round(np.mean([x[0] for x in di_list]), 4),
        'avg_di_torus'  : round(np.mean([x[1] for x in di_list]), 4),
        'avg_hp_normal' : round(np.mean([x[0] for x in hp_list]), 1),
        'avg_hp_torus'  : round(np.mean([x[1] for x in hp_list]), 1),
        'all_di_normal' : [x[0] for x in di_list],
        'all_di_torus'  : [x[1] for x in di_list],
    }

# ── SUMMARY TABLE ─────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Grid':<10} {'Normal DI':>12} {'Toroidal DI':>14}")
print("-"*40)
for gs in GRID_SIZES:
    r = size_results[gs]
    print(f"{str(gs)+'×'+str(gs):<10} "
        f"{r['avg_di_normal']:>12.4f} "
        f"{r['avg_di_torus']:>14.4f}")
print("="*60)
print("patch_size = 4 fixed across all grids — DI values are directly comparable")

# ── PLOT: Bar chart + Scatter ──────────────────────────────────
labels    = [f"{gs}×{gs}" for gs in GRID_SIZES]
di_normal = [size_results[gs]['avg_di_normal'] for gs in GRID_SIZES]
di_torus  = [size_results[gs]['avg_di_torus']  for gs in GRID_SIZES]

x     = np.arange(len(labels))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart
bars1 = axes[0].bar(x - width/2, di_normal, width,
                    label='Normal Grid', color='#2196F3', edgecolor='black')
bars2 = axes[0].bar(x + width/2, di_torus,  width,
                    label='Toroidal Grid', color='#FF5722', edgecolor='black')

for bar in bars1:
    v = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
for bar in bars2:
    v = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=12)
axes[0].set_xlabel("Grid Size", fontsize=12)
axes[0].set_ylabel("Average DI at Convergence", fontsize=12)
axes[0].set_ylim(0, 1.0)
axes[0].set_title("Normal vs Toroidal — Effect of Grid Size on DI\n(patch size fixed at 4×4)", fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=0.6, color='gray', linestyle='--', alpha=0.4)

# Scatter — trial variability
colors_n = '#2196F3'
colors_t = '#FF5722'

for i, gs in enumerate(GRID_SIZES):
    axes[1].scatter(
        [i - 0.15] * TRIALS_A,
        size_results[gs]['all_di_normal'],
        color=colors_n, s=90, alpha=0.8,
        label='Normal' if i == 0 else ""
    )
    axes[1].scatter(
        [i + 0.15] * TRIALS_A,
        size_results[gs]['all_di_torus'],
        color=colors_t, s=90, alpha=0.8,
        label='Toroidal' if i == 0 else ""
    )
    axes[1].plot(
        [i - 0.35, i],
        [np.mean(size_results[gs]['all_di_normal'])] * 2,
        color=colors_n, linewidth=2.5
    )
    axes[1].plot(
        [i, i + 0.35],
        [np.mean(size_results[gs]['all_di_torus'])] * 2,
        color=colors_t, linewidth=2.5
    )

axes[1].set_xticks(range(len(GRID_SIZES)))
axes[1].set_xticklabels(labels, fontsize=12)
axes[1].set_xlabel("Grid Size", fontsize=12)
axes[1].set_ylabel("DI per Trial", fontsize=12)
axes[1].set_title(f"Trial-by-Trial Spread\n(line = average, {TRIALS_A} trials each)", fontsize=12)
axes[1].set_ylim(0, 1.0)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle(
    f"EXPERIMENT A — Does City Size Affect Segregation?  (T={T_A}, patch=4×4 fixed)",
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.show()

# ===== EXPERIMENT B — TASK B1: WHO IS TOLERANT? =====
print("\n" + "="*60)
print("EXPERIMENT B — Task B1: Threshold Grid Visualisation")
print("="*60)

# fresh grid for experiment B
g_test = np.random.choice(
    [0, 1, 2], size=(GRID_B, GRID_B),
    p=[EMPTY_FRAC, A_FRAC, B_FRAC]
)

# create threshold grid
tgrid_test = create_threshold_grid(
    g_test, X_INTOLERANT, T_TOLERANT, T_INTOLERANT
)

# count and verify
n_agents = np.sum(g_test != 0)
n_intol  = np.sum(tgrid_test == T_INTOLERANT)
n_tol    = np.sum((tgrid_test == T_TOLERANT) & (g_test != 0))

print(f"Total agents      : {n_agents}")
print(f"Tolerant  (T=0.2) : {n_tol}  ({round(n_tol/n_agents*100, 1)}%)")
print(f"Intolerant(T=0.7) : {n_intol}  ({round(n_intol/n_agents*100, 1)}%)")
print(f"Target            : {X_INTOLERANT*100}% intolerant")

# build visual map — 0=empty, 1=tolerant, 2=intolerant
visual_map = np.zeros((GRID_B, GRID_B))
for row in range(GRID_B):
    for col in range(GRID_B):
        if g_test[row][col] != 0:
            if tgrid_test[row][col] == T_INTOLERANT:
                visual_map[row][col] = 2   # intolerant → red
            else:
                visual_map[row][col] = 1   # tolerant   → green

cmap_thresh = mcolors.ListedColormap(['white', 'green', 'red'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(g_test, cmap=cmap, vmin=0, vmax=2)
ax1.set_title("Agent Grid\n(blue = Group A,  red = Group B)")
ax1.axis('off')

ax2.imshow(visual_map, cmap=cmap_thresh, vmin=0, vmax=2)
ax2.set_title(
    f"Threshold Map\n"
    f"green = tolerant (T=0.2)   red = intolerant (T=0.7)\n"
    f"{X_INTOLERANT*100}% intolerant  →  {n_intol} agents"
)
ax2.axis('off')

plt.suptitle(
    "Experiment B — Task B1: Who is tolerant and who is intolerant?",
    fontsize=13
)
plt.tight_layout()
plt.show()

# ===== EXPERIMENT B — TASK B2: ONE HETEROGENEOUS SIMULATION =====
print("\n" + "="*60)
print("EXPERIMENT B — Task B2: One Hetero Simulation (X=0.2)")
print("="*60)

STEPS_B2 = 100

# fresh grid
g_b2 = np.random.choice(
    [0, 1, 2], size=(GRID_B, GRID_B),
    p=[EMPTY_FRAC, A_FRAC, B_FRAC]
)
tgrid_b2 = create_threshold_grid(
    g_b2, X_INTOLERANT, T_TOLERANT, T_INTOLERANT
)

# save BEFORE copies
g_b2_before      = g_b2.copy()
tgrid_b2_before  = tgrid_b2.copy()

# record DI before
di_before = get_dissimilarity_index(g_b2, patch_size=4)
print(f"DI before simulation : {di_before:.4f}")

# run simulation
for _ in range(STEPS_B2):
    g_b2, tgrid_b2 = run_one_step_hetero(g_b2, tgrid_b2)

# record DI after
di_after = get_dissimilarity_index(g_b2, patch_size=4)
print(f"DI after  simulation : {di_after:.4f}")
print(f"DI change            : +{di_after - di_before:.4f}")

# build visual maps (0=empty, 1=tolerant, 2=intolerant)
def make_visual_map(grid, tgrid):
    vmap = np.zeros(grid.shape)
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                if tgrid[row][col] == T_INTOLERANT:
                    vmap[row][col] = 2
                else:
                    vmap[row][col] = 1
    return vmap

vmap_before = make_visual_map(g_b2_before, tgrid_b2_before)
vmap_after  = make_visual_map(g_b2, tgrid_b2)

cmap_thresh = mcolors.ListedColormap(['white', 'green', 'red'])

# display 2x2: agent grids top row, threshold maps bottom row
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0][0].imshow(g_b2_before, cmap=cmap, vmin=0, vmax=2)
axes[0][0].set_title(f"Agent Grid — Before\nDI = {di_before:.4f}")
axes[0][0].axis('off')

axes[0][1].imshow(g_b2, cmap=cmap, vmin=0, vmax=2)
axes[0][1].set_title(f"Agent Grid — After {STEPS_B2} steps\nDI = {di_after:.4f}")
axes[0][1].axis('off')

axes[1][0].imshow(vmap_before, cmap=cmap_thresh, vmin=0, vmax=2)
axes[1][0].set_title("Threshold Map — Before\ngreen=tolerant  red=intolerant")
axes[1][0].axis('off')

axes[1][1].imshow(vmap_after, cmap=cmap_thresh, vmin=0, vmax=2)
axes[1][1].set_title("Threshold Map — After\ngreen=tolerant  red=intolerant")
axes[1][1].axis('off')

plt.suptitle(
    f"Experiment B — Task B2: Heterogeneous Simulation\n"
    f"X_intolerant={X_INTOLERANT*100}%  |  T_tolerant={T_TOLERANT}  |  T_intolerant={T_INTOLERANT}",
    fontsize=13
)
plt.tight_layout()
plt.show()

# ===== EXPERIMENT B — TASK B3: FIND X* =====
print("\n" + "="*60)
print("EXPERIMENT B — Task B3: Finding X* (the tipping point)")
print("="*60)

x_axis   = []   # will store X% values for plotting
di_axis  = []   # will store average DI for each X

for x in X_VALUES:
    di_runs = []

    for trial in range(TRIALS_B):
        # fresh grid every trial
        g = np.random.choice(
            [0, 1, 2], size=(GRID_B, GRID_B),
            p=[EMPTY_FRAC, A_FRAC, B_FRAC]
        )
        tgrid = create_threshold_grid(g, x, T_TOLERANT, T_INTOLERANT)

        # run to convergence
        for _ in range(STEPS_B):
            g, tgrid = run_one_step_hetero(g, tgrid)

        di_runs.append(get_dissimilarity_index(g, patch_size=4))

    avg_di = round(np.mean(di_runs), 4)
    x_axis.append(round(x * 100, 0))
    di_axis.append(avg_di)

    print(f"  X = {round(x*100):>3}%  →  avg DI = {avg_di:.4f}")

# ── find X* — the biggest jump between consecutive points ──
max_jump = 0
x_star   = 0
for i in range(1, len(di_axis)):
    jump = di_axis[i] - di_axis[i-1]
    if jump > max_jump:
        max_jump = jump
        x_star   = x_axis[i]

print(f"\nLargest DI jump at X = {x_star}%  →  this is X*")
print(f"Jump size : +{max_jump:.4f}")

# ── plot ──
plt.figure(figsize=(10, 6))
plt.plot(x_axis, di_axis,
         color='crimson', linewidth=2.5, marker='o', markersize=8)

# mark X* with a vertical line
plt.axvline(x=x_star, color='navy', linestyle='--', linewidth=1.8,
            label=f'X* = {x_star}% (tipping point)')

plt.xlabel("% of Intolerant Agents (X)", fontsize=13)
plt.ylabel("DI at Convergence", fontsize=13)
plt.title(
    f"Experiment B — Critical Intolerant Fraction X*\n"
    f"T_tolerant={T_TOLERANT}  |  T_intolerant={T_INTOLERANT}  |  "
    f"Grid={GRID_B}×{GRID_B}  |  {TRIALS_B} trials each",
    fontsize=12
)
plt.ylim(0, 1.0)
plt.xlim(-2, 102)
plt.grid(True, alpha=0.4)
plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5,
            label='DI = 0.5 reference')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# ===== EXPERIMENT B — TASK B4: FINE RESOLUTION around X* =====
print("\n" + "="*60)
print("EXPERIMENT B — Task B4: Fine resolution 0% to 40%")
print("="*60)

X_FINE   = [i/100 for i in range(0, 41, 2)]  # 0%, 2%, 4% ... 40%
TRIALS_B4 = 8    # more trials for accuracy

x_fine_axis  = []
di_fine_axis = []

for x in X_FINE:
    di_runs = []
    for trial in range(TRIALS_B4):
        g = np.random.choice(
            [0, 1, 2], size=(GRID_B, GRID_B),
            p=[EMPTY_FRAC, A_FRAC, B_FRAC]
        )
        tgrid = create_threshold_grid(g, x, T_TOLERANT, T_INTOLERANT)
        for _ in range(STEPS_B):
            g, tgrid = run_one_step_hetero(g, tgrid)
        di_runs.append(get_dissimilarity_index(g, patch_size=4))

    avg_di = round(np.mean(di_runs), 4)
    x_fine_axis.append(round(x * 100, 1))
    di_fine_axis.append(avg_di)
    print(f"  X = {round(x*100):>3}%  →  avg DI = {avg_di:.4f}")

# find steepest point
max_jump = 0
x_star_fine = 0
for i in range(1, len(di_fine_axis)):
    jump = di_fine_axis[i] - di_fine_axis[i-1]
    if jump > max_jump:
        max_jump = jump
        x_star_fine = x_fine_axis[i]

print(f"\nSteepest point at X = {x_star_fine}%")
print(f"Jump size : +{max_jump:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(x_fine_axis, di_fine_axis,
         color='crimson', linewidth=2.5, marker='o', markersize=7)
plt.axvline(x=x_star_fine, color='navy', linestyle='--', linewidth=1.8,
            label=f'Steepest point = {x_star_fine}%')
plt.xlabel("% of Intolerant Agents (X)", fontsize=13)
plt.ylabel("DI at Convergence", fontsize=13)
plt.title(
    f"Experiment B — Fine Resolution (0% to 40%, steps of 2%)\n"
    f"T_tolerant={T_TOLERANT}  |  T_intolerant={T_INTOLERANT}  |  "
    f"Grid={GRID_B}×{GRID_B}  |  {TRIALS_B4} trials each",
    fontsize=12
)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# ===== EXPERIMENT C — TASK C1: UNIFORM POPULATION BASELINE =====
print("\n" + "="*60)
print("EXPERIMENT C — Task C1: Uniform Population (everyone T=0.4)")
print("="*60)
print(f"Mean threshold = {T_UNIFORM}")
print(f"Grid = {GRID_C}×{GRID_C}  |  {TRIALS_C} trials  |  {STEPS_C} steps")

di_uniform = []

for trial in range(TRIALS_C):
    g = np.random.choice(
        [0, 1, 2], size=(GRID_C, GRID_C),
        p=[EMPTY_FRAC, A_FRAC, B_FRAC]
    )
    for _ in range(STEPS_C):
        run_one_step(g, T_UNIFORM)   # everyone uses same threshold

    di_val = get_dissimilarity_index(g, patch_size=4)
    di_uniform.append(di_val)
    print(f"  Trial {trial+1:>2} → DI = {di_val:.4f}")

avg_uniform = round(np.mean(di_uniform), 4)
std_uniform = round(np.std(di_uniform), 4)
print(f"\nUniform avg DI : {avg_uniform}  ±{std_uniform}")
print(f"(mean threshold = {T_UNIFORM})")

# ===== EXPERIMENT C — TASK C2: BIMODAL POPULATION =====
print("\n" + "="*60)
print("EXPERIMENT C — Task C2: Bimodal Population")
print(f"50% agents at T={T_LOW}  +  50% agents at T={T_HIGH}")
print(f"Mean threshold = {(T_LOW + T_HIGH) / 2}  (same as uniform)")
print(f"Grid = {GRID_C}×{GRID_C}  |  {TRIALS_C} trials  |  {STEPS_C} steps")
print("="*60)

di_bimodal = []

for trial in range(TRIALS_C):
    g = np.random.choice(
        [0, 1, 2], size=(GRID_C, GRID_C),
        p=[EMPTY_FRAC, A_FRAC, B_FRAC]
    )
    # create bimodal threshold grid
    tgrid = create_bimodal_threshold_grid(g, T_LOW, T_HIGH)

    for _ in range(STEPS_C):
        g, tgrid = run_one_step_hetero(g, tgrid)

    di_val = get_dissimilarity_index(g, patch_size=4)
    di_bimodal.append(di_val)
    print(f"  Trial {trial+1:>2} → DI = {di_val:.4f}")

avg_bimodal = round(np.mean(di_bimodal), 4)
std_bimodal = round(np.std(di_bimodal), 4)
print(f"\nBimodal avg DI : {avg_bimodal}  ±{std_bimodal}")
print(f"(mean threshold = {(T_LOW + T_HIGH)/2})")

# ===== COMPARISON =====
print("\n" + "="*60)
print("EXPERIMENT C — COMPARISON SUMMARY")
print("="*60)
print(f"Uniform  (T=0.4 for all)       → avg DI = {avg_uniform}  ±{std_uniform}")
print(f"Bimodal  (50% T=0.1, 50% T=0.7) → avg DI = {avg_bimodal}  ±{std_bimodal}")
diff = round(avg_bimodal - avg_uniform, 4)
print(f"Difference                       → {'+' if diff > 0 else ''}{diff}")
if diff > 0:
    print("→ Bimodal produces MORE segregation than uniform (same mean)")
    print("→ Granovetter 1978 CONFIRMED in Schelling ABM")
elif diff < 0:
    print("→ Bimodal produces LESS segregation than uniform")
else:
    print("→ No difference detected")

# ===== BAR CHART =====
fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(
    ['Uniform\n(T=0.4 for all)', f'Bimodal\n(50% T={T_LOW}, 50% T={T_HIGH})'],
    [avg_uniform, avg_bimodal],
    color=['#2196F3', '#E53935'],
    width=0.4,
    edgecolor='black'
)

# error bars
ax.errorbar(
    [0, 1], [avg_uniform, avg_bimodal],
    yerr=[std_uniform, std_bimodal],
    fmt='none', color='black', capsize=8, linewidth=2
)

# value labels on bars
for bar, val in zip(bars, [avg_uniform, avg_bimodal]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
            f'{val:.4f}', ha='center', fontsize=13, fontweight='bold')

ax.set_ylabel("Average DI at Convergence", fontsize=13)
ax.set_title(
    f"Experiment C — Same Mean Threshold (0.4), Different Distribution\n"
    f"Grid={GRID_C}×{GRID_C}  |  {TRIALS_C} trials  |  {STEPS_C} steps",
    fontsize=12
)
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, label='DI = 0.5')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ===== EXPERIMENT C — VISUAL COMPARISON (1 trial each) =====
print("\nRunning visual comparison trial...")

# ── Uniform — one trial ──────────────────────────
g_uni = np.random.choice(
    [0,1,2], size=(GRID_C, GRID_C),
    p=[EMPTY_FRAC, A_FRAC, B_FRAC]
)
di_uni_time = [get_dissimilarity_index(g_uni, patch_size=4)]

for _ in range(STEPS_C):
    run_one_step(g_uni, T_UNIFORM)
    di_uni_time.append(get_dissimilarity_index(g_uni, patch_size=4))

# ── Bimodal — one trial ──────────────────────────
g_bi = np.random.choice(
    [0,1,2], size=(GRID_C, GRID_C),
    p=[EMPTY_FRAC, A_FRAC, B_FRAC]
)
tgrid_bi = create_bimodal_threshold_grid(g_bi, T_LOW, T_HIGH)
di_bi_time = [get_dissimilarity_index(g_bi, patch_size=4)]

for _ in range(STEPS_C):
    g_bi, tgrid_bi = run_one_step_hetero(g_bi, tgrid_bi)
    di_bi_time.append(get_dissimilarity_index(g_bi, patch_size=4))

# ── Window 1: final grids side by side ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(g_uni, cmap=cmap, vmin=0, vmax=2)
axes[0].set_title(
    f"Uniform Population — Final Grid\n"
    f"Everyone T={T_UNIFORM}  |  DI = {di_uni_time[-1]:.4f}",
    fontsize=12
)
axes[0].axis('off')

axes[1].imshow(g_bi, cmap=cmap, vmin=0, vmax=2)
axes[1].set_title(
    f"Bimodal Population — Final Grid\n"
    f"50% T={T_LOW}, 50% T={T_HIGH}  |  DI = {di_bi_time[-1]:.4f}",
    fontsize=12
)
axes[1].axis('off')

plt.suptitle(
    f"Experiment C — Same Mean Threshold (0.4), Different Distribution\n"
    f"Grid={GRID_C}×{GRID_C}  |  {STEPS_C} steps",
    fontsize=13
)
plt.tight_layout()
plt.show()

# ── Window 2: DI over time both on same graph ───
plt.figure(figsize=(10, 6))
plt.plot(di_uni_time, color='#2196F3', linewidth=2.5,
         label=f'Uniform (T={T_UNIFORM} for all)  →  final DI={di_uni_time[-1]:.4f}')
plt.plot(di_bi_time, color='#E53935', linewidth=2.5,
         label=f'Bimodal (50% T={T_LOW}, 50% T={T_HIGH})  →  final DI={di_bi_time[-1]:.4f}')
plt.xlabel("Simulation Step", fontsize=13)
plt.ylabel("Dissimilarity Index (DI)", fontsize=13)
plt.title(
    f"Experiment C — DI over time: Uniform vs Bimodal\n"
    f"Same mean threshold (0.4)  |  Grid={GRID_C}×{GRID_C}",
    fontsize=12
)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()