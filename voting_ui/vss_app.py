"""Pedersen VSS: монитор с честным разделением публичного и приватного доступа.

Запуск из корня репозитория:
    python -m streamlit run voting_ui/vss_app.py

В режиме участника данные берутся только из обычных методов Board.
read_all() используется ИСКЛЮЧИТЕЛЬНО в диагностическом God View.
Это учебная симуляция, без аутентификации и реального шифрования.
"""
from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threshold_voting.vss_protocol import VSSVotingProtocol
from src.core.board import (PrivateShare, PublicCommitments,
                            ParticipantTotalShare, MaliciousReport)

st.set_page_config(page_title="ConfidentialVoting · приватные доли", layout="wide")
st.title("ConfidentialVoting · Pedersen VSS")
st.caption("fᵢ(j) видит только получатель; sⱼ = F(j) доступны только участникам раунда; "
           "обязательства и MaliciousReport публичны.")
st.warning("Модель доступа учебная: шифрование и подтверждение личности ещё не подключены. "
           "Настоящий веб-доступ других пользователей этим экраном НЕ разграничивается.")

with st.sidebar:
    st.header("Параметры симуляции")
    n = st.slider("Участников", 2, 10, 5)
    variants = st.slider("Вариантов OHE", 2, 8, 3)
    t = st.slider("Порог t", 2, n, min(3, n))
    p = st.number_input("p: p и 2p+1 — простые", min_value=n+1, value=23, step=1)
    options = [f"Вариант {i+1}" for i in range(variants)]
    choices = tuple(st.selectbox(f"Выбор участника {i}", range(variants),
                                  format_func=lambda k: options[k],
                                  index=(i-1) % variants, key=f"vss_choice_{i}_{variants}")
                    for i in range(1, n+1))
    scenario = st.selectbox("Сценарий", (
        "honest", "invalid_vote", "wrong_private_share", "wrong_total_share",
        "false_complaint_changed", "false_complaint_valid"))
    attacker = st.selectbox("Malicious-участник", range(1, n+1), index=min(1, n-1))
    receiver = st.selectbox("Получатель приватной доли", [k for k in range(1, n+1) if k != attacker])
    delta = st.number_input("Искажение Δ", min_value=1, max_value=max(1, int(p)-1), value=7)
    launch = st.button("Запустить симуляцию", type="primary")

if launch:
    try:
        protocol = VSSVotingProtocol(
            choices, variants, t, int(p), scenario=scenario,
            attacker=attacker, receiver=receiver, delta=int(delta))
        result = protocol.run()
        st.session_state["vss_protocol_private"] = protocol
        st.session_state["vss_result_private"] = result
    except Exception as exc:
        st.session_state.pop("vss_protocol_private", None)
        st.session_state.pop("vss_result_private", None)
        st.error(f"Ошибка запуска: {type(exc).__name__}: {exc}")

protocol = st.session_state.get("vss_protocol_private")
result = st.session_state.get("vss_result_private")
if protocol is None or result is None:
    st.info("Задай параметры и нажми «Запустить симуляцию».")
    st.stop()

st.subheader("Режим просмотра")
mode = st.radio("Отображение", (
    "Публичное: commitments и reports",
    "Участник: мои fᵢ(j) и общие sⱼ",
    "God View: все данные (отладка)",
), horizontal=True)

st.subheader("Состояние голосования")
a, b, c, d = st.columns(4)
a.metric("Статус", result.status)
b.metric("Активны", ", ".join(map(str, result.active_ids)) or "нет")
c.metric("Исключены", ", ".join(map(str, result.excluded_ids)) or "нет")
d.metric("Текущий порог t", result.threshold)
if result.counts is None:
    st.warning("Итог не сформирован: " + result.status)
elif mode.startswith("Публичное"):
    st.info("Суммарный OHE-итог вычислен участниками, но не публиковался наружу; "
            "для его просмотра переключись в модель участника или God View.")
else:
    st.success("Итог OHE, восстановленный участниками: " + str(result.counts))
    st.bar_chart({f"Вариант {k+1}": count for k, count in enumerate(result.counts)})

st.subheader("Проверка malicious")
rows = []
for round_audit in result.rounds:
    if not round_audit.verdicts:
        rows.append({"Раунд": round_audit.round_id, "Этап": round_audit.phase,
                     "Репортёр": "—", "Обвиняемый": "—", "Вердикт": "нет жалоб",
                     "Исключён": "—"})
    for verdict in round_audit.verdicts:
        rows.append({"Раунд": round_audit.round_id, "Этап": round_audit.phase,
                     "Репортёр": verdict.report.reporter, "Обвиняемый": verdict.report.suspect,
                     "Вердикт": verdict.reason, "Исключён": verdict.guilty_id})
st.dataframe(rows, use_container_width=True, hide_index=True)

st.subheader("BulletinBoard — уровни видимости")

round_audit = st.selectbox("Раунд", result.rounds,
                           format_func=lambda item: f"#{item.round_id} · {item.phase}")
round_id = round_audit.round_id

# Каждый режим использует собственный способ чтения Board.
# Нельзя сначала read_all(), а затем лишь визуально скрыть чужие доли.
items: list[tuple[str, int | None, int | None, object]] = []
if mode.startswith("God View"):
    st.error("Только локальная диагностика! Этот режим раскрывает чужие приватные доли.")
    for entry in protocol.board.read_all():
        if entry.message.round_id == round_id:
            msg = entry.message
            items.append((type(msg).__name__, getattr(msg, "sender", getattr(msg, "reporter", None)),
                          getattr(msg, "receiver", None), msg))
else:
    # Общедоступные сведения: ни f_i(j), ни агрегированных s_j в этом наборе нет.
    for msg in protocol.board.get_commitments(round_id):
        items.append(("PublicCommitments", msg.sender, None, msg))
    for msg in protocol.board.get_malicious_reports(round_id):
        items.append(("MaliciousReport", msg.reporter, msg.suspect, msg))
    if mode.startswith("Участник"):
        me = st.selectbox("Показывать от имени участника", round_audit.active_before)
        # В раунде список имел фиксированный состав. Новые excluded ID
        # не получают доступ к долям следующих раундов.
        for msg in protocol.board.get_private_shares(me, round_id):
            items.append(("PrivateShare (мой входящий)", msg.sender, msg.receiver, msg))
        for msg in protocol.board.get_total_shares(me, round_id):
            items.append(("ParticipantTotalShare (для голосующих)", msg.sender, None, msg))

st.dataframe([{"Тип": typ, "От": sender, "Кому": receiver,
               "Содержимое": str(payload)}
              for typ, sender, receiver, payload in items],
             use_container_width=True, hide_index=True)

if mode.startswith("Публичное"):
    st.info("В публичном просмотре отсутствуют исходные приватные доли и "
            "агрегированные sⱼ. Жалобы могут раскрывать конкретную fᵢ(j) как доказательство.")
elif mode.startswith("Участник"):
    st.info("fᵢ(j) других получателей не показаны; sⱼ доступны только как "
            "сведения раунда для голосующих.")
st.caption("Ограничения: Pedersen VSS не доказывает one-hot допустимость секрета "
           "без ZKP; God View и параметры симуляции не предназначены для реальных голосующих.")
