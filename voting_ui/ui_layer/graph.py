"""Graphviz DOT граф передачи долей; источником значений служит Frame."""
from __future__ import annotations
from math import cos, sin, tau
from .projections import Frame


def build_dot(frame: Frame) -> str:
    """Не выводить v_i на рёбрах, не раскрывать скрытые f_i(j)."""
    n = frame.n
    sent = {(d.sender, d.receiver): d.value for d in frame.visible_deals}
    published = {d.sender for d in frame.board_deals}
    members = set(frame.coalition) if frame.mode != 'god' else set(range(1, n + 1))
    lines = [
        'digraph G {',
        '  graph [layout=neato, overlap=false, splines=true, outputorder=edgesfirst, bgcolor="transparent", pad=0.35];',
        '  node [shape=circle, style="filled", fontname="Arial", fontsize=13, penwidth=1.7, fixedsize=true, width=0.85];',
        '  edge [fontname="Arial", fontsize=10, arrowsize=0.55];',
    ]
    radius = 3.25
    for j in range(1, n + 1):
        angle = tau * (j - 1) / n + 1.5707963267948966
        x, y = radius * cos(angle), radius * sin(angle)
        active = j == frame.active
        fill = '#dbeafe' if j in members else '#f1f5f9'
        border = '#2563eb' if active else ('#334155' if j in members else '#cbd5e1')
        lines.append(f'  P{j} [label="P{j}", pos="{x:.3f},{y:.3f}!", pin=true, fillcolor="{fill}", color="{border}"];')
    lines.append('  BOARD [label="BOARD", shape=box, width=1.15, height=0.60, pos="0,0!", pin=true, fillcolor="#fef3c7", color="#d97706"];')

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            deal_step = i + 1
            visible = (i, j) in sent
            is_current = deal_step == frame.step
            if visible:
                color, style = ('#2563eb', 'solid') if is_current else ('#94a3b8', 'solid')
                label = f'f_{i}({j})={sent[i,j]}'
                penwidth = 2.3 if is_current else 1.15
            else:
                # Никаких чисел: неизвестное или ещё не отправленное сообщение.
                color, style, label, penwidth = '#e2e8f0', 'dashed', '', 0.7
            lines.append(f'  P{i} -> P{j} [color="{color}", style={style}, label="{label}", penwidth={penwidth}];')

    for j in range(1, n + 1):
        pub_step = n + 2 + j
        published_here = j in published
        value = sent.get((j, -1))
        is_current = pub_step == frame.step
        if published_here and value is not None:
            label = f's_{j}={value}'
            color = '#059669' if is_current else '#86efac'
            style, penwidth = 'solid', 2.2 if is_current else 1.2
        else:
            label, color, style, penwidth = '', '#e2e8f0', 'dashed', 0.75
        lines.append(f'  P{j} -> BOARD [color="{color}", style={style}, label="{label}", penwidth={penwidth}];')
    lines.append('}')
    return '\n'.join(lines)
