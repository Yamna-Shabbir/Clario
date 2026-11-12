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
    *,
    decision_graph: Optional[Dict[str, Any]] = None,
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

    if decision_graph:
        for node in decision_graph.get("nodes") or []:
            node_id = node.get("id")
            if not node_id:
                continue
            G.add_node(
                node_id,
                type=str(node.get("type", "")).lower(),
                label=node.get("label") or node_id,
            )
        for edge in decision_graph.get("edges") or []:
            source = edge.get("source")
            target = edge.get("target")
            if source in G and target in G:
                G.add_edge(source, target, relation=edge.get("relation", ""))
        if not G.nodes:
            G.add_node("meeting", type="meeting", label="Meeting")
    else:
        _add_task_graph(G, tasks)

    # ── Colour map ─────────────────────────────────────────────────────────────
    color_map = []
    for node in G.nodes:
        node_type = G.nodes[node].get("type", "")
        if node_type == "meeting":
            color_map.append("#263A35")     # forest
        elif node_type == "task":
            color_map.append("#829B8A")     # sage
        elif node_type == "person":
            color_map.append("#6A8274")     # sage deep
        elif node_type == "decision":
            color_map.append("#DCE8DF")     # mint soft
        else:                               # deadline / date
            color_map.append("#8B6348")     # warm accent

    # ── Layout and draw ────────────────────────────────────────────────────────
    print("Building knowledge graph …")
    pos = nx.spring_layout(G, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=1500, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, ax=ax)
    edge_labels = {
        (src, dst): data.get("relation", "")
        for src, dst, data in G.edges(data=True)
        if data.get("relation")
    }
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="#F7F5EF", ax=ax)

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


def _add_task_graph(G: Any, tasks: List[Dict[str, Any]]) -> None:
    """Original task-only graph used when no decision graph is supplied."""
    MEETING_NODE = "Meeting"
    G.add_node(MEETING_NODE, type="meeting", label=MEETING_NODE)

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
    colors = ['#263A35', '#829B8A', '#DCE8DF', '#6A8274', '#829B8A', '#263A35']
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]
    
    bars = ax.bar(names, counts, color=bar_colors, edgecolor='none', width=0.6, alpha=0.85)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#263A35')

    ax.set_ylabel("Number of Tasks", fontweight='bold', color='#5F726B')
    ax.set_title("Tasks by Assignee", fontsize=14, fontweight='bold', pad=15, color='#263A35')
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#829B8A')
    ax.spines['bottom'].set_color('#829B8A')
    
    plt.xticks(rotation=45 if len(names) > 4 else 0, ha='right' if len(names) > 4 else 'center')
    ax.tick_params(colors='#5F726B')
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

    color_map = {"High": "#8B6348", "Medium": "#829B8A", "Low": "#DCE8DF", "No tasks": "#C5D4CB"}
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
        wedgeprops=dict(width=0.4, edgecolor='#FFFFFF', linewidth=2),
        textprops=dict(color="#263A35", fontweight='bold', fontsize=9)
    )
    
    # Improve label readability
    for text in texts:
        text.set_color('#263A35')
        text.set_fontsize(10)

    ax.set_title("Task Priority Distribution", fontsize=14, fontweight='bold', pad=15, color='#263A35')
    
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

    known = ("pending", "started", "doing", "completed")
    by_status = {name: 0 for name in known}
    for task in tasks:
        st = task.get("status", "pending")
        if st not in by_status:
            st = "pending"
        by_status[st] += 1

    filtered_status = {k: v for k, v in by_status.items() if v > 0}
    label_map = {
        "pending": "Pending",
        "started": "Started",
        "doing": "Doing",
        "completed": "Completed",
    }
    if not filtered_status:
        labels = ["No tasks"]
        sizes = [1]
        pie_colors = ["#7A7A85"]
    else:
        labels = [label_map[k] for k in filtered_status.keys()]
        sizes = list(filtered_status.values())
        color_map = {
            "Completed": "#829B8A",
            "Pending": "#DCE8DF",
            "Started": "#6A8274",
            "Doing": "#8B6348",
        }
        pie_colors = [color_map.get(l, "#C5D4CB") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=pie_colors, 
        wedgeprops=dict(width=0.6, edgecolor='#FFFFFF', linewidth=2),
        textprops=dict(color="#263A35", fontweight='bold', fontsize=9)
    )
    
    for text in texts:
        text.set_color('#263A35')
        text.set_fontsize(10)

    ax.set_title("Task Completion Status", fontsize=14, fontweight='bold', pad=15, color='#263A35')
    
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
    colors = ['#829B8A', '#DCE8DF']
    
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts = ax.pie(
        sizes, 
        colors=colors, 
        startangle=90, 
        wedgeprops=dict(width=0.3, edgecolor='#FFFFFF', linewidth=2)
    )

    # Add center text
    percentage = int(ratio * 100)
    plt.text(0, 0.1, f"{percentage}%", ha='center', va='center', fontsize=32, fontweight='bold', color='#263A35')
    plt.text(0, -0.2, "Productivity\nScore", ha='center', va='center', fontsize=12, color='#5F726B')

    ax.set_title("Meeting Actionability Index", color="#263A35", pad=20, fontsize=14, fontweight='bold')
    
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

    fig, ax = plt.subplots(figsize=(7, 5))
    
    bars = ax.barh(labels, values, color='#829B8A', edgecolor='none', height=0.6)

    ax.tick_params(colors='#5F726B')
    for spine in ax.spines.values():
        spine.set_edgecolor('#829B8A')

    ax.set_title("Task Timeline Bottleneck", color="#263A35", pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Tasks", color="#5F726B", labelpad=10)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    va='center', color='#263A35', fontweight='bold')

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
    
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.grid(color='#DCE8DF', linestyle='--', alpha=0.9)

    for spine in ax.spines.values():
        spine.set_edgecolor('#829B8A')

    status_labels = ["Pending", "Completed"]
    priority_labels = ["Low", "Medium", "High"]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(status_labels, color='#5F726B')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(priority_labels, color='#5F726B')
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 2.5)

    # Plot bubbles
    for (x, y), count in counts.items():
        # Color based on priority (y)
        color = '#8B6348' if y == 2 else ('#829B8A' if y == 1 else '#DCE8DF')
        ax.scatter(x, y, s=count * 1000, color=color, alpha=0.85, edgecolors='#FFFFFF', linewidth=2)
        ax.text(x, y, str(count), color='#263A35', fontweight='bold', fontsize=12, ha='center', va='center')

    ax.set_title("Risk & Execution Matrix", color="#263A35", pad=20, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()
