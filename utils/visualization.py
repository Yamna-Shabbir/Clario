"""
utils/visualization.py
-----------------------
Knowledge graph and chart generation utilities.
Extracted from Clario Project notebook (cell 115).
"""

import os
import json
from typing import List, Dict, Any, Optional


def build_knowledge_graph(
    tasks: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Build and render a knowledge-graph visualisation of the detected tasks.

    Creates a directed graph where:
      - Each task node is labelled with a short description snippet.
      - Person nodes are connected to the tasks assigned to them.
      - Date nodes are attached where a due_date was extracted.

    Parameters
    ----------
    tasks       : List of task dicts (description, assigned_to, due_date, keyword).
    output_path : If given, save the PNG to this path.  Defaults to
                  ``data/graphs/knowledge_graph.png`` relative to the project root.
    show        : Whether to call ``plt.show()`` after rendering (interactive use).

    Returns
    -------
    str – Absolute path of the saved image, or an empty string if saving was
          skipped and ``show=False``.
    """
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")           # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "visualization.py requires 'networkx' and 'matplotlib'. "
            f"Install them first.  Original error: {exc}"
        )

    # ── Build graph ────────────────────────────────────────────────────────────
    G = nx.DiGraph()

    MEETING_NODE = "Meeting"
    G.add_node(MEETING_NODE, type="meeting")

    for task in tasks:
        desc    = task.get("description", "")
        person  = task.get("assigned_to", "Unknown")
        due     = task.get("due_date",    "Not specified")
        keyword = task.get("keyword",     "")
        task_id = f"Task-{task.get('id', len(G.nodes))}"

        # Shorten label
        label = desc[:50] + ("…" if len(desc) > 50 else "")
        G.add_node(task_id, type="task", label=label, keyword=keyword)
        G.add_edge(MEETING_NODE, task_id)

        if person != "Unknown":
            G.add_node(person, type="person")
            G.add_edge(task_id, person)

        if due != "Not specified":
            due_node = f" {due}"
            G.add_node(due_node, type="date")
            G.add_edge(task_id, due_node)

    # ── Colour map ─────────────────────────────────────────────────────────────
    color_map = []
    for node in G.nodes:
        node_type = G.nodes[node].get("type", "")
        if node_type == "meeting":
            color_map.append("#3B4A6B")     # blue
        elif node_type == "task":
            color_map.append("#C48B47")     # amber
        elif node_type == "person":
            color_map.append("#4A7C59")     # green
        else:                               # date
            color_map.append("#A64444")     # red

    # ── Layout and draw ────────────────────────────────────────────────────────
    print("Building knowledge graph …")
    pos = nx.spring_layout(G, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=1500, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, ax=ax)

    labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="#EAECEF", ax=ax)

    ax.set_title("️ Clario Knowledge Graph", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    # ── Save ───────────────────────────────────────────────────────────────────
    saved_path = ""
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        saved_path = os.path.abspath(output_path)
        print(f" Knowledge graph saved to: {saved_path}")

    if show:
        plt.show()

    plt.close(fig)
    return saved_path


def generate_assignee_bar_chart(
    tasks: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Generate a professional bar chart showing tasks per person.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(f"visualization.py requires 'matplotlib'. Error: {exc}")

    by_person = {}
    for task in tasks:
        person = task.get("assigned_to", "Unknown")
        by_person[person] = by_person.get(person, 0) + 1

    if not by_person:
        by_person["No tasks"] = 0

    names = list(by_person.keys())
    counts = list(by_person.values())

    # Styling
    if 'seaborn-v0_8-whitegrid' in plt.style.available:
        plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Beautiful colors
    colors = ['#3B4A6B', '#4A7C59', '#C48B47', '#A64444', '#7A7A85', '#4A6D7C']
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]
    
    bars = ax.bar(names, counts, color=bar_colors, edgecolor='none', width=0.6, alpha=0.85)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#EAECEF')

    ax.set_ylabel("Number of Tasks", fontweight='bold', color='#B0B4BE')
    ax.set_title("Tasks by Assignee", fontsize=14, fontweight='bold', pad=15, color='#EAECEF')
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#3A3D45')
    ax.spines['bottom'].set_color('#3A3D45')
    
    plt.xticks(rotation=45 if len(names) > 4 else 0, ha='right' if len(names) > 4 else 'center')
    ax.tick_params(colors='#B0B4BE')
    plt.tight_layout()

    saved_path = ""
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
        saved_path = os.path.abspath(output_path)

    if show:
        plt.show()

    plt.close(fig)
    return saved_path


def generate_priority_donut_chart(
    tasks: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Generate a professional donut chart showing task priority distribution.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(f"visualization.py requires 'matplotlib'. Error: {exc}")

    by_priority = {"High": 0, "Medium": 0, "Low": 0}
    for task in tasks:
        prio = task.get("priority", "Low")
        if prio in by_priority:
            by_priority[prio] += 1
        else:
            by_priority["Low"] += 1

    # Filter out zero counts
    filtered_priority = {k: v for k, v in by_priority.items() if v > 0}

    color_map = {"High": "#A64444", "Medium": "#C48B47", "Low": "#4A7C59", "No tasks": "#7A7A85"}
    if not filtered_priority:
        filtered_priority = {"No tasks": 1}

    labels = list(filtered_priority.keys())
    sizes = list(filtered_priority.values())

    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Priority semantic colors
    pie_colors = [color_map.get(l, "#7A7A85") for l in labels]
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=pie_colors, 
        wedgeprops=dict(width=0.4, edgecolor='#12151C', linewidth=2),
        textprops=dict(color="w", fontweight='bold', fontsize=9)
    )
    
    # Improve label readability
    for text in texts:
        text.set_color('#EAECEF')
        text.set_fontsize(10)

    ax.set_title("Task Priority Distribution", fontsize=14, fontweight='bold', pad=15, color='#EAECEF')
    
    plt.tight_layout()

    saved_path = ""
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
        saved_path = os.path.abspath(output_path)

    if show:
        plt.show()

    plt.close(fig)
    return saved_path


def generate_status_pie_chart(
    tasks: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Generate a professional pie chart showing pending vs completed tasks.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(f"visualization.py requires 'matplotlib'. Error: {exc}")

    by_status = {"pending": 0, "completed": 0}
    for task in tasks:
        st = task.get("status", "pending")
        if st in by_status:
            by_status[st] += 1
        else:
            by_status["pending"] += 1

    filtered_status = {k: v for k, v in by_status.items() if v > 0}
    if not filtered_status:
        labels = ["No tasks"]
        sizes = [1]
        pie_colors = ["#7A7A85"]
    else:
        labels = ["Completed" if k == "completed" else "Pending" for k in filtered_status.keys()]
        sizes = list(filtered_status.values())
        # Semantic colors for status
        color_map = {"Completed": "#4A7C59", "Pending": "#C48B47"}
        pie_colors = [color_map.get(l, "#7A7A85") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=pie_colors, 
        wedgeprops=dict(width=0.6, edgecolor='#12151C', linewidth=2),
        textprops=dict(color="w", fontweight='bold', fontsize=9)
    )
    
    for text in texts:
        text.set_color('#EAECEF')
        text.set_fontsize(10)

    ax.set_title("Task Completion Status", fontsize=14, fontweight='bold', pad=15, color='#EAECEF')
    
    plt.tight_layout()

    saved_path = ""
    if output_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
        saved_path = os.path.abspath(output_path)

    if show:
        plt.show()

    plt.close(fig)
    return saved_path

def generate_actionability_index(total_sentences: int, total_tasks: int, output_path: str) -> None:
    """
    Generates a Donut Chart representing the Actionability Index (Productivity Score) of the meeting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if total_sentences <= 0:
        total_sentences = 1

    ratio = total_tasks / total_sentences
    # Cap ratio visually at 100% just in case
    ratio = min(ratio, 1.0)
    
    # We plot the ratio vs the remaining
    sizes = [ratio, 1.0 - ratio]
    colors = ['#C48B47', '#3B4A6B']
    
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0A0C10")
    wedges, texts = ax.pie(
        sizes, 
        colors=colors, 
        startangle=90, 
        wedgeprops=dict(width=0.3, edgecolor='#0A0C10', linewidth=2)
    )

    # Add center text
    percentage = int(ratio * 100)
    plt.text(0, 0.1, f"{percentage}%", ha='center', va='center', fontsize=32, fontweight='bold', color='#EAECEF')
    plt.text(0, -0.2, "Productivity\nScore", ha='center', va='center', fontsize=12, color='#B0B4BE')

    ax.set_title("Meeting Actionability Index", color="#EAECEF", pad=20, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()

def generate_timeline_chart(tasks: list, output_path: str) -> None:
    """
    Generates a horizontal bar chart visualizing tasks by their extracted due dates.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    due_dates = [t.get("due_date", "Not specified").title() for t in tasks]
    counts = Counter(due_dates)

    # Sort so 'Not specified' is at the bottom if present
    labels = sorted(counts.keys(), key=lambda x: (1 if x.lower() == 'not specified' else 0, x))
    values = [counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0A0C10")
    
    # Premium colors
    bars = ax.barh(labels, values, color='#4A7C59', edgecolor='#0A0C10', height=0.6)

    ax.set_facecolor("#0A0C10")
    ax.tick_params(colors='#B0B4BE')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3B4A6B')

    ax.set_title("Task Timeline Bottleneck", color="#EAECEF", pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Tasks", color="#B0B4BE", labelpad=10)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    va='center', color='#EAECEF', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()

def generate_risk_matrix(tasks: list, output_path: str) -> None:
    """
    Generates a Bubble Chart (Matrix) mapping Priority vs Status.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    # X axis: Status, Y axis: Priority
    status_map = {"pending": 0, "completed": 1}
    priority_map = {"Low": 0, "Medium": 1, "High": 2}
    
    points = []
    for t in tasks:
        st = t.get("status", "pending").lower()
        pr = t.get("priority", "Low").title()
        if st in status_map and pr in priority_map:
            points.append((status_map[st], priority_map[pr]))
            
    counts = Counter(points)
    
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0A0C10")
    ax.set_facecolor("#0A0C10")

    # Draw grid
    ax.grid(color='#3B4A6B', linestyle='--', alpha=0.5)

    for spine in ax.spines.values():
        spine.set_edgecolor('#3B4A6B')

    status_labels = ["Pending", "Completed"]
    priority_labels = ["Low", "Medium", "High"]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(status_labels, color='#B0B4BE')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(priority_labels, color='#B0B4BE')
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 2.5)

    # Plot bubbles
    for (x, y), count in counts.items():
        # Color based on priority (y)
        color = '#A64444' if y == 2 else ('#C48B47' if y == 1 else '#4A7C59')
        ax.scatter(x, y, s=count * 1000, color=color, alpha=0.8, edgecolors='#0A0C10', linewidth=2)
        ax.text(x, y, str(count), color='#EAECEF', fontweight='bold', fontsize=12, ha='center', va='center')

    ax.set_title("Risk & Execution Matrix", color="#EAECEF", pad=20, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()
