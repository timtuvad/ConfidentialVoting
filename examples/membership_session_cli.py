"""Пример работы отдельных классов без Streamlit и симуляции Шамира."""
from src.core.board import InMemoryBoard, MaliciousReport
from src.core.voting import VotingSession

board = InMemoryBoard()
session = VotingSession(threshold=4, board=board)
session.admission.register_many((1, 2, 3, 4, 5))
session.start()
session.begin_reporting(1)
for reporter in session.active_ids:
    for suspect in session.active_ids:
        if suspect != reporter:
            board.post_malicious_report(MaliciousReport(1, reporter, suspect, suspect in (2, 3)))
transition = session.process_reports(1)
print('Action:', transition.action.value)   # restart
print('Active:', transition.active_ids)     # (1, 4, 5)
print('New t:', transition.current_threshold)  # 3
print('Events:', session.registry.events)
# На действие RESTART внешний движок обязан создать и разослать новые полиномы.
