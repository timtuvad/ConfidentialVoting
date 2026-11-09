"""Единый Streamlit-интерфейс ConfidentialVoting на базе Pedersen VSS.

Запуск из корня репозитория:
    python -m streamlit run voting_ui/app.py

Только локальная учебная симуляция! Секретные доли пока НЕ шифруются,
а выбор роли в UI НЕ является аутентификацией. Не публикуйте приложение
в Интернет с реальными голосами/долями.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.board import (
    MaliciousReport, ParticipantTotalShare, PrivateShare,
    PublicCommitments, ReportKind,
)
from src.core.utils import is_prime
from threshold_voting.vss_protocol import VSSVotingProtocol

st.set_page_config(page_title="ConfidentialVoting · OHE + Pedersen VSS", page_icon="🔐", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.4rem; max-width: 1580px}
[data-testid="stMetricValue"] {font-size: 1.5rem}
</style>""", unsafe_allow_html=True)

SCENARIOS = {
    "honest": "✅ Honest — корректное голосование",
    "invalid_vote": "🔴 Недопустимый OHE-вектор (VSS не обнаруживает)",
    "wrong_private_share": "🔴 Подмена приватной fᵢ(j)",
    "wrong_total_share": "🔴 Подмена агрегированной sⱼ = F(j)",
    "false_complaint_changed": "🔴 Ложная жалоба с изменённой долей",
    "false_complaint_valid": "🔴 Ложная жалоба на корректную долю",
}
STAGE_NAMES = {
    0: "Этап 1 · Публичные обязательства",
    1: "Этап 1 · Передача fᵢ(j)",
    2: "Этап 1 · Разбор жалоб",
    3: "Этап 2 · Обмен суммарными sⱼ",
    4: "Этап 2 · Проверка sⱼ и жалобы",
}


@dataclass(frozen=True)
class TimelineEvent:
    stage: int
    round_id: int
    message_id: int | None
    kind: str
    sender: int
    receiver: int | None
    payload: Any


def stage_for(message: object) -> int:
    if isinstance(message, PublicCommitments):
        return 0
    if isinstance(message, PrivateShare):
        return 1
    if isinstance(message, ParticipantTotalShare):
        return 3
    if isinstance(message, MaliciousReport):
        if message.kind == ReportKind.PRIVATE_SHARE:
            return 2
        return 4
    raise TypeError(f"Неподдерживаемый тип записи Board: {type(message).__name__}")


def as_event(message: object, message_id: int | None = None) -> TimelineEvent:
    stage = stage_for(message)
    sender = message.reporter if isinstance(message, MaliciousReport) else message.sender
    receiver = (message.receiver if isinstance(message, PrivateShare)
                else message.suspect if isinstance(message, MaliciousReport) else None)
    return TimelineEvent(stage, message.round_id, message_id, type(message).__name__,
                         sender, receiver, message)


def visible_events(board: Any, round_id: int, mode: str,
                   viewer_id: int | tuple[int, ...] | None = None) -> tuple[TimelineEvent, ...]:
    """Не использовать read_all() в публичном и участническом режиме.

    Снимок порядка ограниченных выборок сгруппирован по этапам, поскольку
    обычные методы Board не раскрывают глобальный sequence_number.
    """
    if mode == "god":
        return tuple(as_event(entry.message, entry.message_id)
                     for entry in board.read_all() if entry.message.round_id == round_id)

    messages = [*board.get_commitments(round_id),
                *board.get_malicious_reports(round_id)]
    if mode in ("participant", "coalition"):
        viewers = ((viewer_id,) if isinstance(viewer_id, int)
                   else tuple(viewer_id) if viewer_id is not None else ())
        if not viewers or (mode == "participant" and len(viewers) != 1):
            raise ValueError("Нужно выбрать одного или нескольких голосующих")
        # Объединение *разрешённых* приватных входящих сообщений членов коалиции.
        # Без системного read_all(): Board проверяет каждого viewer отдельно.
        for voter in viewers:
            messages.extend(board.get_private_shares(voter, round_id))
        # Все голосующие видят одинаковые агрегированные доли.
        messages.extend(board.get_total_shares(viewers[0], round_id))

    return tuple(as_event(item) for item in sorted(messages, key=stage_for))


def format_bundle(bundle: tuple) -> str:
    """Печатает только доли, уже разрешённые выбранному режиму."""
    parts = []
    for v in bundle:
        parts.append(f"(x={v.x.value}, f={v.value.value}, r={v.blinding_share.value})")
    return "[" + ", ".join(parts) + "]"


def describe(event: TimelineEvent) -> tuple[str, str]:
    m = event.payload
    if isinstance(m, PublicCommitments):
        lines = []
        for k, c in enumerate(m.value):
            coeff_commitments = tuple(x.value for x in c.commitment_vector)
            lines.append(f"k={k}: {coeff_commitments}")
        return f"P{m.sender} опубликовал PolynomialCommitments", "\n".join(lines)
    if isinstance(m, PrivateShare):
        return (f"P{m.sender} → P{m.receiver}: fᵢ(j)", format_bundle(m.value))
    if isinstance(m, ParticipantTotalShare):
        return (f"P{m.sender} передал участникам sⱼ = F(j)", format_bundle(m.value))
    if isinstance(m, MaliciousReport):
        kind = "fᵢ(j)" if m.kind == ReportKind.PRIVATE_SHARE else "sⱼ"
        detail = f"source_message_id={m.source_message_id}"
        if m.revealed_share is not None:
            detail += " · раскрытие в жалобе: " + format_bundle(m.revealed_share)
        return (f"P{m.reporter} обвиняет P{m.suspect} · {kind}", detail)
    raise TypeError(type(m))


def graph_dot(events: tuple[TimelineEvent, ...], active: tuple[int, ...],
              excluded: tuple[int, ...], current: TimelineEvent | None) -> str:
    """DOT граф Board. Только события, уже разрешённые выбранному режиму."""
    all_ids = sorted(set(active) | set(excluded))
    out = [
        "digraph Voting {",
        'graph [rankdir=LR, bgcolor="transparent", pad="0.2", splines=true];',
        'node [fontname="Helvetica", style=filled, shape=circle];',
        'edge [fontname="Helvetica", fontsize=9, arrowsize=0.65];',
        'board [label="BOARD", shape=box, fillcolor="#fef3c7", color="#d97706"];',
    ]
    for pid in all_ids:
        fill = "#fee2e2" if pid in excluded else "#dbeafe"
        out.append(f'p{pid} [label="P{pid}", fillcolor="{fill}"];')
    for idx, event in enumerate(events):
        m = event.payload
        highlight = event == current and current is not None
        color = "#dc2626" if isinstance(m, MaliciousReport) else (
            "#2563eb" if highlight else "#94a3b8")
        # Рёбра показывают только факт доступного события; значения — в таблице.
        if isinstance(m, PrivateShare):
            src, dst, label = f"p{m.sender}", f"p{m.receiver}", "fᵢ(j)"
        elif isinstance(m, PublicCommitments):
            src, dst, label = f"p{m.sender}", "board", "C"
        elif isinstance(m, ParticipantTotalShare):
            src, dst, label = f"p{m.sender}", "board", "sⱼ"
        elif isinstance(m, MaliciousReport):
            src, dst, label = f"p{m.reporter}", "board", f"report P{m.suspect}"
        else:
            continue
        out.append(f'{src} -> {dst} [label="{label}", color="{color}", '
                   f'penwidth={2.5 if highlight else 1.0}, key="msg_{idx}"];')
    out.append("}")
    return "\n".join(out)


def event_table(events: tuple[TimelineEvent, ...], current_idx: int | None = None) -> list[dict]:
    result = []
    for idx, event in enumerate(events):
        title, detail = describe(event)
        result.append({
            "Шаг": idx + 1,
            "Этап": STAGE_NAMES[event.stage],
            "От": f"P{event.sender}",
            "Кому / suspect": f"P{event.receiver}" if event.receiver else "—",
            "Сообщение": title,
            "Содержимое": detail,
            "Текущий": "◀" if idx == current_idx else "",
        })
    return result


def render_summary(result: Any, options: tuple[str, ...], mode: str,
                   viewer_id: int | tuple[int, ...] | None = None) -> None:
    st.subheader("Состояние голосования")
    a, b, c, d = st.columns(4)
    a.metric("Статус", result.status)
    b.metric("Активные", ", ".join(map(str, result.active_ids)) or "нет")
    c.metric("Исключённые", ", ".join(map(str, result.excluded_ids)) or "нет")
    d.metric("Текущий t", result.threshold)
    if result.status != "complete" or result.counts is None:
        st.warning("Достоверный итог не сформирован (" + result.status + ").")
    elif mode == "public":
        st.info("Итог OHE не публиковался для посторонних: его видят только голосующие.")
    elif (mode == "participant" and viewer_id not in result.active_ids or
          mode == "coalition" and not (set(viewer_id or ()) & set(result.active_ids))):
        st.info("Выбранные участники не входят в окончательный состав голосующих. "
                "Итог последнего раунда для них недоступен.")
    else:
        st.success("Итог OHE: " + ", ".join(f"{name} — {count}"
                                          for name, count in zip(options, result.counts)))
        st.bar_chart({name: result.counts[k] for k, name in enumerate(options)})
        top = max(result.counts)
        winners = [options[k] for k, v in enumerate(result.counts) if v == top]
        st.caption("Победитель: " + ", ".join(winners)
                   + (" (ничья)" if len(winners) > 1 else ""))


def render_rounds(result: Any) -> None:
    st.subheader("Раунды и исключения")
    st.dataframe([{
        "Раунд": row.round_id,
        "Порог t": row.threshold,
        "Участники в начале": ", ".join(map(str, row.active_before)),
        "Исключены к концу": ", ".join(map(str, row.excluded_after_round)) or "—",
        "Фаза": row.phase,
        "Жалоб": len(row.verdicts),
    } for row in result.rounds], hide_index=True, use_container_width=True)
    verdict_rows = []
    for row in result.rounds:
        for v in row.verdicts:
            verdict_rows.append({
                "Раунд": row.round_id,
                "Жалобщик": f"P{v.report.reporter}",
                "Подозреваемый": f"P{v.report.suspect}",
                "Сообщение": v.report.source_message_id,
                "Решение": v.reason,
                "Виновный": "—" if v.guilty_id is None else f"P{v.guilty_id}",
            })
    if verdict_rows:
        st.dataframe(verdict_rows, hide_index=True, use_container_width=True)
    else:
        st.info("MaliciousReport не поступали.")


def run_app() -> None:
    st.title("🔐 ConfidentialVoting · OHE + Pedersen VSS")
    st.caption("Единый интерфейс: ShamirCryptography, PolynomialCommitments, "
               "VerifiableShare, Board, MaliciousReport и восстановление OHE.")
    st.warning("Только учебная локальная симуляция. Нет настоящего шифрования, "
               "аутентификации и разграничения веб-сессий. Параметры симуляции "
               "доступны владельцу приложения и раскрывают выборы участников.")

    with st.sidebar:
        st.header("⚙️ Параметры голосования")
        n = st.slider("Участников n", min_value=2, max_value=10, value=5)
        variants_text = st.text_input("Варианты через запятую", "Альфа, Бета, Гамма")
        options = tuple(x.strip() for x in variants_text.split(",") if x.strip())
        if not 2 <= len(options) <= 10 or len(set(options)) != len(options):
            st.error("Нужно от 2 до 10 различных названий вариантов.")
            st.stop()
        t = st.slider("Порог Шамира t", min_value=2, max_value=n, value=min(n, 3))
        p = int(st.number_input("Специальное простое p", min_value=n+1,
                                value=max(23, n + 1), step=1,
                                help="p и q=2p+1 обязаны быть простыми. p>n."))
        st.caption(f"q = 2p + 1 = {2*p+1}")
        if not (is_prime(p) and is_prime(2*p+1) and p > n and p > 2):
            st.error("Выбери p>n, для которого p и 2p+1 простые. Например, 23.")
            st.stop()
        st.subheader("Голоса (управление симуляцией)")
        choices = tuple(options.index(st.selectbox(
            f"P{i}", options, index=(i-1) % len(options),
            key=f"unified_choice_{n}_{i}_{len(options)}")) for i in range(1, n+1))
        st.divider()
        st.subheader("Malicious-сценарий")
        scenario = st.selectbox("Поведение", tuple(SCENARIOS), format_func=SCENARIOS.get)
        attacker = int(st.selectbox("Атакующий", range(1, n+1), index=min(1, n-1)))
        receiver = int(st.selectbox("Получатель fᵢ(j)",
                                    tuple(i for i in range(1, n+1) if i != attacker)))
        delta = int(st.number_input("Δ по модулю p", min_value=1, max_value=p-1,
                                    value=min(7, p-1)))
        max_rounds = int(st.number_input("Максимум раундов", min_value=1,
                                         max_value=30, value=10))
        launch = st.button("▶ Запустить новый расчёт", type="primary",
                           use_container_width=True)

    if launch:
        try:
            protocol = VSSVotingProtocol(choices, len(options), t, p=p,
                                         scenario=scenario, attacker=attacker,
                                         receiver=receiver, delta=delta,
                                         max_rounds=max_rounds)
            result = protocol.run()
            st.session_state["unified_protocol"] = protocol
            st.session_state["unified_result"] = result
            st.session_state["unified_options"] = options
            st.session_state["unified_scenario"] = scenario
        except Exception as err:
            for key in ("unified_protocol", "unified_result", "unified_options"):
                st.session_state.pop(key, None)
            st.exception(err)

    protocol = st.session_state.get("unified_protocol")
    result = st.session_state.get("unified_result")
    if protocol is None or result is None:
        st.info("Укажи параметры и нажми «Запустить новый расчёт».")
        st.stop()

    options = st.session_state["unified_options"]
    st.subheader("🔎 Режим наблюдателя")
    mode_label = st.radio("Доступ к информации", (
        "Публичный наблюдатель", "Участник голосования",
        "Коалиция участников", "God View (диагностика)"
    ), horizontal=True)
    mode = ("public" if mode_label.startswith("Публичный") else
            "participant" if mode_label.startswith("Участник") else
            "coalition" if mode_label.startswith("Коалиция") else "god")
    if mode == "god":
        st.error("God View раскрывает все доли. Доступен только в локальной учебной модели.")

    if st.session_state.get("unified_scenario") == "invalid_vote":
        st.warning("Недопустимый OHE-секрет может пройти все проверки Pedersen VSS: "
                   "для допустимости бюллетеня нужен отдельный ZKP.")

    if not result.rounds:
        render_summary(result, options, mode)
        st.info("Нет раундов для отображения.")
        return
    round_item = st.selectbox("Раунд", result.rounds, key="unified_round",
                              format_func=lambda r: f"#{r.round_id} · t={r.threshold} · {r.phase}")
    viewer_id = None
    if mode == "participant":
        viewer_id = st.selectbox("Смотреть от имени участника",
                                 round_item.active_before, key=f"unified_viewer_{round_item.round_id}")
        st.caption("Можно читать только входящие fᵢ(j) и доступные этому раунду sⱼ. "
                   "В реальном приложении личность должна задаваться аутентификацией.")
    elif mode == "coalition":
        viewer_id = tuple(st.multiselect("Участники коалиции", round_item.active_before,
                        default=round_item.active_before[:min(2, len(round_item.active_before))],
                        key=f"unified_coalition_{round_item.round_id}"))
        if not viewer_id:
            st.info("Добавь хотя бы одного участника коалиции.")
            return
        st.caption("Коалиция видит объединение *своих входящих* fᵢ(j) и общие sⱼ. "
                   "При ≥t членах могут раскрыться индивидуальные секреты дилеров.")

    render_summary(result, options, mode, viewer_id)
    render_rounds(result)
    st.subheader("📜 Пошаговый протокол и BulletinBoard")
    try:
        events = visible_events(protocol.board, round_item.round_id, mode, viewer_id)
    except (PermissionError, ValueError) as err:
        st.error(f"Доступ запрещён: {err}")
        return

    step = st.slider("Сколько доступных сообщений уже показано", 0, len(events),
                     len(events), key=f"unified_step_{round_item.round_id}_{mode}_{viewer_id}")
    shown = events[:step]
    current = shown[-1] if shown else None
    if current is not None:
        st.info(f"Шаг {step}/{len(events)} · {STAGE_NAMES[current.stage]} · "
                f"{describe(current)[0]}")
    else:
        st.info("Начало выбранного раунда (до появления сообщений).")

    left, right = st.columns([1.05, 1.45])
    with left:
        st.markdown("**Граф участников и Board**")
        try:
            st.graphviz_chart(graph_dot(shown, round_item.active_before,
                                        round_item.excluded_after_round, current),
                              use_container_width=True)
        except Exception as err:
            st.warning(f"Graphviz недоступен: {err}")
        st.caption("Граф показывает только доступные сообщения. "
                   "Красное ребро — жалоба, голубое — текущий шаг.")
    with right:
        st.markdown("**Содержимое доступных записей**")
        st.dataframe(event_table(shown, step - 1 if step else None),
                     hide_index=True, use_container_width=True)

    if mode == "public":
        st.caption("Публично видны только обязательства и MaliciousReport. "
                   "Жалоба может намеренно раскрывать конкретный VerifiableShare.")
    elif mode in ("participant", "coalition"):
        st.caption("Чужие приватные доли не запрашивались: показаны только входящие "
                   "доли выбранного участника/коалиции. Агрегированные sⱼ доступны "
                   "участникам выбранного раунда.")
    else:
        st.caption("God View вызвал read_all() ради отладки. "
                   "Этот доступ нельзя предоставлять реальным пользователям.")
    if mode == "god":
        with st.expander("Секреты бюллетеней — только диагностический режим"):
            st.write({f"P{pid}": tuple(v) for pid, v in protocol.ballots.items()})


if __name__ == "__main__":
    run_app()
