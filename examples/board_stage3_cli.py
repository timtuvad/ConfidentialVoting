"""Пример Board + MaliciousReport + restart без Streamlit.

Запуск: python -m examples.board_stage3_cli
"""
from threshold_voting.consensus import manual_detector
from threshold_voting.simulation import Simulation
from src.core.board import PrivateShare, PublicShare, MaliciousReport

sim = Simulation(
    n=5, t=4, p=101, seed=17,
    choices=(0, 1, 2, 1, 3), options=('A', 'B', 'C', 'D'),
    enable_consensus=True,
    detector=manual_detector({1: (2, 3), 4: (2, 3), 5: (2, 3)}),
)

last = sim.snapshots[-1]
print('Статус:', last.consensus_status)
print('Активные:', last.active_participants)
print('Текущий порог:', last.current_threshold)
print('Итог:', last.final_result)
for event in sim.board.read_all():
    kind = type(event.message).__name__
    msg = event.message
    print(f'#{event.message_id:02d} round={msg.round_id}: {kind}')
