import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set font family to serif
plt.rcParams['font.family'] = 'serif'

csv_data = """Model,avg_wnp,avg_wnp_rank,prediction_rate,prediction_rate_rank,match_rate,match_rate_rank,,,
anthropic/claude-haiku-4.5,36.75%,1,100.00%,1,49.41%,15,,,
google/gemini-3-flash-preview,36.82%,2,100.00%,1,96.27%,1,,,
openai/gpt-5.2,38.42%,11,96.06%,11,83.21%,3,,,
x-ai/grok-4-fast,37.47%,7,99.75%,3,79.15%,4,,,
deepseek/deepseek-chat-v3-0324,38.72%,13,98.41%,4,70.69%,9,,,
minimax/minimax-m2.1,38.62%,12,94.36%,13,60.94%,12,,,
moonshotai/kimi-k2-thinking,39.17%,14,97.51%,5,47.39%,16,,,
xiaomi/mimo-v2-flash,37.01%,4,96.48%,9,65.50%,10,,,
google/gemma-3-12b-it,37.24%,6,96.76%,7,58.33%,13,,,
google/gemma-3-27b-it,37.85%,9,95.86%,12,62.15%,11,,,
mistralai/ministral-14b-2512,39.57%,15,96.60%,8,77.02%,6,,,
qwen/qwen3-30b-a3b-instruct-2507,37.17%,5,97.35%,6,57.76%,14,,,
Qwen/Qwen3-1.7B,40.39%,16,91.24%,15,73.15%,8,,,
Qwen/Qwen3-4B-Instruct-2507,37.95%,10,86.11%,16,74.19%,7,,,
GIM-1.7B,37.47%,8,93.87%,14,77.04%,5,,,
GIM-4B,36.84%,3,96.35%,10,83.78%,2,,,
"""

# Load data
df = pd.read_csv(io.StringIO(csv_data))

# Select relevant columns and clean index
cols = ['Model', 'avg_wnp', 'avg_wnp_rank', 'prediction_rate', 'prediction_rate_rank', 'match_rate', 'match_rate_rank']
df = df[cols]
# Shorten model names by removing the organization prefix
df['Model'] = df['Model'].apply(lambda x: x.split('/')[-1][:13]+"..." if len(x.split('/')[-1]) > 13 else x.split('/')[-1])
df.set_index('Model', inplace=True)

# Create figure
# Simulating a single-column width for ACM/KDD (approx 3.3-4 inches wide)
# Adjusted figsize to be "skinnier" and shorter to look like a compact table
# Reduced width further to minimal to reduce horizontal padding inside cells
fig, axes = plt.subplots(1, 3, figsize=(3.2, 3.5), sharey=True)

# Define column pairs (Value Column, Rank Column)
pairs = [
    ('avg_wnp', 'avg_wnp_rank'),
    ('prediction_rate', 'prediction_rate_rank'),
    ('match_rate', 'match_rate_rank')
]

# Create custom heatmaps
# Using 'Greens_r' for a clean, classic academic Green look
# Rank 1 (Best) -> Dark Green, Rank 16 (Worst) -> Light/Pale Green
cmap = 'Greens_r'

for i, (val_col, rank_col) in enumerate(pairs):
    ax = axes[i]
    
    # We want to plot the Rank as the color value
    # We need to reshape the series into a DataFrame for heatmap
    rank_data = df[[rank_col]]
    
    # We want to annotate with the actual Value strings
    # User requested Rank in small parentheses
    # Using Subscript MathText to render rank smaller: Value_{(Rank)}
    annot_data = df.apply(lambda row: f"{row[val_col]}$_{{({row[rank_col]})}}$", axis=1).to_frame()
    
    sns.heatmap(
        rank_data, 
        annot=annot_data, 
        fmt="", 
        cmap=cmap, 
        ax=ax, 
        cbar=False,
        linewidths=0.5, 
        linecolor='white',
        annot_kws={"size": 7} # Smaller font for compact view
    )

    # Booktabs style lines
    # Midrules inside data (4, 8, 12, 14) are thin (0.5)
    # Bottom rule (16) is thick (2.0)
    boundary_lines = [
        (4, 0.5),
        (8, 0.5),
        (12, 0.5),
        (14, 0.5),
        (16, 2.0)
    ]
    
    for y, lw in boundary_lines:
        xmin = 0
        xmax = 1
        if i == 0:
            # Extend line to the left to cover labels
            xmin = -1.5 
            
        ax.hlines(y, xmin, xmax, color='black', linewidth=lw, clip_on=False)
        
    # Add Top Rule (above headers) and Header Rule (below headers)
    # Since headers are axes titles/ticks, valid y coordinates are tricky.
    # We use axes fraction coordinates to draw lines above/below the plotting area.
    # Top rule at very top, Header rule just above data.
    
    # Header Rule (the one separating headers from data) - roughly at y=0
    xmin, xmax = 0, 1
    if i == 0: xmin = -1.5
    ax.hlines(0, xmin, xmax, color='black', linewidth=0.5, clip_on=False) # Midrule style for header separator

    # Top Rule (above the text)
    # We need to draw this in a way that goes above the x-tick labels
    # Using relative coordinates might be best, but hlines works in data coordinates.
    # Data stats at 0. negative y is up. Let's try drawing at y=-1 or similar?
    # No, ticks are outside. Let's use ax.annotate or transform.
    # Actually, simpler: draw a line at y=-1.5 (guesswork for height of text)
    if i == 0: xmin = -1.5
    ax.hlines(-1.2, xmin, xmax, color='black', linewidth=2.0, clip_on=False) 

    
    # Titles and Labels
    # Clean up title and add arrows
    if 'avg_wnp' in val_col:
        title_text = r'Avg. WNP $\downarrow$'
    elif 'prediction_rate' in val_col:
        title_text = r'Pred. Rate $\uparrow$'
    elif 'match_rate' in val_col:
        title_text = r'Match Rate $\uparrow$'
    else:
        title_text = val_col
    
    # Use x-ticks at the top instead of title
    ax.set_xticks([0.5])
    ax.set_xticklabels([title_text])
    ax.xaxis.tick_top()
    # Remove x-tick lines
    ax.tick_params(axis='x', length=0)
    
    # Rotate top labels to match model format
    # User requested no rotation for x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=7)
    
    ax.set_xlabel('')
    
    # Remove y-tick marks (lines), keeping labels only for the first plot
    ax.tick_params(axis='y', length=0)

    # Adjust y-axis label
    ax.set_ylabel('') # Remove 'Model' label completely
    
    if i == 0:
        # User requested horizontal labels, left aligned
        # Using ha='left' and a large pad to simulate left alignment column
        ax.set_yticklabels(df.index, rotation=0, ha='left', fontsize=7)
        ax.tick_params(axis='y', pad=60) 
    else:
        ax.set_ylabel('')

# Add Legend (Colorbar) for Rank
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Adjust layout to make room for colorbar at the bottom
plt.tight_layout(rect=[0, 0.08, 1, 1]) 
plt.subplots_adjust(wspace=0) # Remove spacing between subplots

# Create Colorbar
norm = Normalize(vmin=1, vmax=16)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add colorbar axis [left, bottom, width, height] in figure fraction
cbar_ax = fig.add_axes([0.3, 0.04, 0.4, 0.02]) 
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
# Move label to the side (right) to save vertical space
cbar_ax.text(1.05, 0.5, 'Rank', transform=cbar_ax.transAxes, 
             fontsize=7, va='center', ha='left')
# cbar.set_label('Rank', fontsize=7, labelpad=2)
cbar.ax.tick_params(labelsize=6, length=2)
cbar.set_ticks([1, 4, 8, 12, 16]) # Show ticks for clarity

plt.savefig('model_heatmap.pdf', format='pdf', bbox_inches='tight')
plt.savefig('model_heatmap.png', dpi=300, bbox_inches='tight') # Also save high-res PNG for preview