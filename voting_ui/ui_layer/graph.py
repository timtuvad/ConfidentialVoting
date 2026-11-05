"""Graphviz-представление векторных f_i(j) из Frame, без доступа к ядру."""
from __future__ import annotations
from math import sin, cos, tau
from .projections import Frame


def _vec(values: tuple[int, ...]) -> str:
    return '[' + ','.join(map(str, values)) + ']'


def build_dot(frame: Frame) -> str:
    sent = {(d.sender, d.receiver):d.value for d in frame.visible_deals}
    private = set(frame.coalition) if frame.mode != 'god' else set(range(1, frame.n+1))
    info = frame.attack_info or {}
    lines = [
        'digraph G {',
        'graph [layout=neato, overlap=false, splines=true, outputorder=edgesfirst, bgcolor="transparent", pad=0.3];',
        'node [shape=circle, style=filled, fontname="Arial", fixedsize=true, width=0.75];',
        'edge [fontname="Arial", fontsize=9, arrowsize=0.52];',
    ]
    radius = 3.5
    for i in range(1, frame.n+1):
        angle = tau*(i-1)/frame.n
        x, y = radius*cos(angle), radius*sin(angle)
        fill = '#dbeafe' if i in private else '#f1f5f9'
        color = '#2563eb' if i==frame.active else '#94a3b8'
        lines.append(f'P{i} [label="P{i}", pos="{x:.2f},{y:.2f}!", pin=true, fillcolor="{fill}", color="{color}"];')
    lines.append('BOARD [label="BOARD", shape=box, pos="0,0!", pin=true, fillcolor="#fef3c7", color="#d97706"];')
    for i in range(1, frame.n+1):
        for j in range(1, frame.n+1):
            if i==j: continue
            v=sent.get((i,j))
            current=(i+1==frame.step)
            color='#2563eb' if current else '#94a3b8'
            if v is None: color='#e2e8f0'
            if (info.get('type')=='wrong_private_share' and i==info.get('attacker')
                    and j==info.get('receiver') and v is not None): color='#dc2626'
            label='' if v is None else f'f_{i}({j})={_vec(v)}'
            style='dashed' if v is None else 'solid'
            lines.append(f'P{i} -> P{j} [color="{color}", style={style}, label="{label}"];')
    for j in range(1,frame.n+1):
        value=sent.get((j,-1))
        current=(frame.n+2+j==frame.step)
        color='#059669' if current else '#86efac'
        if value is None: color='#e2e8f0'
        if (info.get('type')=='wrong_public_share' and info.get('attacker')==j
                and value is not None): color='#dc2626'
        label='' if value is None else f's_{j}={_vec(value)}'
        style='dashed' if value is None else 'solid'
        lines.append(f'P{j} -> BOARD [color="{color}", style={style}, label="{label}"];')
    return '\n'.join(lines+['}'])
