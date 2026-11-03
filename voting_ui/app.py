"""Streamlit-отладчик порогового голосования.

Запуск: streamlit run app.py
UI получает только Snapshot через source.py -> projections.py -> Frame.
Внутренние объекты Participant и BulletinBoard здесь не используются.
"""
from __future__ import annotations

from itertools import combinations
import hashlib

import streamlit as st

from ui_layer.contracts import Trace
from ui_layer.graph import build_dot
from ui_layer.observer import alternative_curves, horner, reconstruct_public
from ui_layer.projections import Frame, project
from ui_layer.source import load_backend, load_demo
from ui_layer.contracts import trace_from_json


st.set_page_config(page_title='ConfidentialVoting · Debugger', page_icon='🔐', layout='wide')
st.markdown('''
<style>
.block-container {padding-top: 1.4rem; max-width: 1600px;}
.hint {color:#64748b; font-size:.88rem;}
.step-header {font-size:1.13rem; font-weight:650; padding:.5rem 0;}
[data-testid="stMetricValue"] {font-size:1.45rem;}
</style>''', unsafe_allow_html=True)


def polynomial_text(poly: tuple[int, ...] | None, idx: int) -> str:
    """Только форматирование уже разрешённых коэффициентов."""
    if poly is None:
        return f'f_{idx}(x) = ?'
    pieces = [str(poly[0])]
    for degree, coefficient in enumerate(poly[1:], 1):
        pieces.append(f'{coefficient}x' if degree == 1 else f'{coefficient}x^{degree}')
    return f'f_{idx}(x) = ' + ' + '.join(pieces)


def load_selected_trace() -> tuple[Trace | None, str]:
    """Загрузка демо/JSON/движка; никаких вычислений над секретами."""
    with st.sidebar:
        st.header('⚙️ Источник данных')
        src = st.radio('Режим', ['Демо', 'Загрузить JSON', 'Simulation (Python)'],
                       help='Демо и JSON работают без ещё не написанного протокольного ядра.')
        if src == 'Демо':
            st.info('Готовые корректные снимки: n=5, t=3, p=101, seed=42. Данные не пересчитываются.')
            return load_demo(), 'demo'
        if src == 'Загрузить JSON':
            uploaded = st.file_uploader('Выберите trace.json', type=['json'])
            if uploaded is None:
                st.info('Загрузите снимки в формате, описанном в README_UI.md.')
                return None, 'upload:none'
            content = uploaded.getvalue()
            return trace_from_json(content), 'upload:' + hashlib.sha256(content).hexdigest()

        n = st.number_input('Количество участников n', min_value=2, max_value=10, value=5)
        t = st.number_input('Порог t', min_value=2, max_value=int(n), value=min(3, int(n)))
        p = st.number_input('Простое p (>n)', min_value=int(n) + 1, value=101)
        seed = st.number_input('Seed', value=42, step=1)
        st.write('Голоса участников')
        votes = tuple(int(st.checkbox(f'Участник {j}: голос 1', value=(j in (1, 3, 4)),
                                      key=f'vote_{j}')) for j in range(1, int(n) + 1))
        scenario = st.selectbox('Сценарий', ['honest', 'coalition', 'substitution'])
        if st.button('Создать снимки', type='primary', use_container_width=True):
            try:
                trace = load_backend(int(n), int(t), int(p), int(seed), votes, scenario)
            except (ImportError, ModuleNotFoundError, TypeError, ValueError, NotImplementedError) as exc:
                st.error(f'Не удалось подключить Simulation: {exc}')
                return None, 'backend:unavailable'
            st.session_state['backend_trace'] = trace
            st.session_state['backend_key'] = repr((n, t, p, seed, votes, scenario))
        trace = st.session_state.get('backend_trace')
        if trace is None:
            st.warning('В текущем репозитории отсутствует threshold_voting.simulation.Simulation. '
                       'Используйте демо или JSON до реализации протокола.')
            return None, 'backend:none'
        return trace, 'backend:' + st.session_state['backend_key']


def render_cards(frame: Frame) -> None:
    st.subheader('Участники')
    for card in frame.cards:
        name = f'P{card.idx}'
        icon = '🔓' if card.can_see_private else '🔒'
        status = 'опубликовано' if card.published else 'не опубликовано'
        with st.expander(f'{icon} {name}  ·  {status}', expanded=(card.idx == frame.active)):
            a, b = st.columns(2)
            a.metric('Голос vᵢ', '?' if card.vote is None else str(card.vote))
            b.metric('Доля sᵢ', '?' if card.share is None else str(card.share))
            st.code(polynomial_text(card.poly, card.idx), language=None)
            if not card.can_see_private:
                st.caption('Чужие входящие доли недоступны в этом режиме.')
            elif not card.received:
                st.caption('Входящих долей пока нет.')
            else:
                st.table([{'От участника': i, 'Значение fᵢ(j)': value}
                          for i, value in card.received])


def render_invariants(frame: Frame) -> None:
    st.subheader('Инварианты I1–I7')
    st.caption('Статусы поступают от тестера, а не вычисляются интерфейсом. '
               'В ограниченном режиме приватные диагностики скрыты; I6 требует отдельного теста.')
    names = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7']
    legend = {'ok': '🟢', 'warning': '🟡', 'fail': '🔴', 'pending': '⏳',
              'hidden': '🔒', 'unknown': '⚪'}
    columns = st.columns(7)
    for col, name in zip(columns, names):
        value = frame.checks.get(name, 'unknown')
        label = legend.get(value, '⚪')
        with col:
            st.markdown(f'**{name}**  {label}')
            st.caption({'ok':'OK', 'warning':'Коллизия', 'fail':'Ошибка',
                        'pending':'Позже', 'hidden':'Скрыто', 'unknown':'Не проверен'}
                       .get(value, value))


def render_reconstruction(frame: Frame) -> None:
    st.subheader('Реконструкция по публичным долям')
    board = {d.sender: d.value for d in frame.board_deals}
    available = sorted(board)
    if not available:
        st.info('Пока никто не опубликовал sⱼ на доске.')
        return
    selected = st.multiselect('Выбрать участников, опубликовавших sⱼ', available,
                              default=available[:min(frame.t, len(available))],
                              key=f'reconstruction_{frame.step}')
    selected = sorted(selected)
    public_points = {j: board[j] for j in selected}
    if len(selected) == frame.t:
        value = reconstruct_public(public_points, frame.t, frame.p)
        st.success(f'По {frame.t} публичным долям восстановлено **S = F(0) = {value}**')
        st.caption('Значение вычислено независимым наблюдателем, не BulletinBoard.')
    elif len(selected) < frame.t:
        st.warning(f'Выбрано {len(selected)} < t={frame.t}: эти точки сами по себе '
                   f'не определяют F(0). Возможны все {frame.p} значений в Fₚ.')
        first, second = alternative_curves(public_points, frame.t, frame.p)
        st.code(f'F₀(x): коэффициенты {first}  → F₀(0)=0\n'
                f'F₁(x): коэффициенты {second}  → F₁(0)=1', language=None)
        xs = list(range(frame.n + 1))
        st.line_chart({'F₀(x)': [horner(first, x, frame.p) for x in xs],
                       'F₁(x)': [horner(second, x, frame.p) for x in xs]})
        if len(available) >= frame.t:
            st.caption('Другие уже опубликованные доли позволяют восстановить итог, '
                       'но выбранное подмножество — нет.')
    else:
        st.warning('Выбрано больше t долей. Для сравнения комбинаций используйте ровно t.')

    if len(available) == frame.n and frame.step == frame.total_steps:
        with st.expander(f'Проверка всех C({frame.n},{frame.t}) комбинаций', expanded=True):
            rows = []
            for subset in combinations(available, frame.t):
                value = reconstruct_public({j: board[j] for j in subset}, frame.t, frame.p)
                rows.append({'Доли': ', '.join(map(str, subset)), 'S': value})
            st.dataframe(rows, hide_index=True, use_container_width=True)
            values = {row['S'] for row in rows}
            (st.success if len(values) == 1 else st.error)(
                'Все подмножества восстановили одинаковый S.' if len(values) == 1
                else 'Ошибка: восстановленные значения не совпадают!')


def main() -> None:
    st.title('🔐 ConfidentialVoting · Protocol Debugger')
    st.caption('Внешний интерфейс на Snapshot. Протокол Шамира и участники находятся вне UI.')
    try:
        trace, source_key = load_selected_trace()
    except (KeyError, ValueError, TypeError) as exc:
        st.error(f'Некорректный формат Snapshot: {exc}')
        st.stop()
    if trace is None:
        st.stop()

    # Смена трассы сбрасывает только положение курсора, не параметры движка.
    if st.session_state.get('trace_key') != source_key:
        st.session_state['step'] = 1
        st.session_state['trace_key'] = source_key
    st.session_state['step'] = max(1, min(st.session_state['step'], len(trace.snapshots)))

    st.markdown(f'**Сценарий:** `{trace.scenario}`')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Участников n', trace.n)
    c2.metric('Порог t', trace.t)
    c3.metric('Поле Fₚ', trace.p)
    c4.metric('Seed', trace.seed)

    left, right = st.columns([1.3, 2.2])
    with left:
        mode = st.radio('View as', ['god', 'participant', 'coalition'], horizontal=True)
    chosen: tuple[int, ...] = ()
    with right:
        if mode == 'participant':
            chosen = (int(st.selectbox('Участник', list(range(1, trace.n + 1)))),)
        elif mode == 'coalition':
            picked = st.multiselect('Члены коалиции C', list(range(1, trace.n + 1)),
                                    default=[1, 2])
            if not picked:
                st.info('Выберите хотя бы одного члена коалиции.')
                st.stop()
            chosen = tuple(sorted(picked))
            if len(chosen) >= trace.t:
                st.warning(f'|C|={len(chosen)} ≥ t: приватные доли других участников '
                           'могут быть восстановлены коалицией.')
    button1, button2, button3, space = st.columns([1, 1, 1, 4])
    with button1:
        if st.button('◀ Назад', disabled=st.session_state['step'] == 1,
                     use_container_width=True):
            st.session_state['step'] -= 1
    with button2:
        if st.button('Вперёд ▶', disabled=st.session_state['step'] == len(trace.snapshots),
                     use_container_width=True):
            st.session_state['step'] += 1
    with button3:
        if st.button('↺ Сбросить', use_container_width=True):
            st.session_state['step'] = 1
    st.slider('Шаг протокола', min_value=1, max_value=len(trace.snapshots), key='step')

    snap = trace.snapshots[st.session_state['step'] - 1]
    frame = project(trace, snap, mode, chosen)
    st.markdown(f'<div class="step-header">Шаг {frame.step}/{frame.total_steps} · {frame.phase.upper()}</div>',
                unsafe_allow_html=True)
    st.info(frame.event)

    graph_col, cards_col = st.columns([1.7, 1.0], gap='large')
    with graph_col:
        st.subheader('Граф обмена долями')
        st.graphviz_chart(build_dot(frame), use_container_width=True)
        st.caption('Синие рёбра — текущие известные сообщения, серые — прошлые, '
                   'пунктир — скрытые или будущие. На рёбрах нет значений vᵢ.')
    with cards_col:
        render_cards(frame)

    st.divider()
    log_col, board_col = st.columns([1.1, 1])
    with log_col:
        st.subheader('Лог событий')
        for prev in trace.snapshots[:frame.step]:
            # Только безопасные метаданные, без обращения к приватным полям снимка.
            from ui_layer.projections import safe_description
            st.text(f'{prev.step:02d}. {safe_description(prev.phase, prev.active, trace.n, trace.t)}')
    with board_col:
        st.subheader('BulletinBoard · read()')
        if frame.board_deals:
            st.dataframe([{'От кого': d.sender, 'Опубликованная доля sⱼ': d.value}
                          for d in frame.board_deals], hide_index=True,
                         use_container_width=True)
        else:
            st.caption('Публичных сообщений пока нет.')
    st.divider()
    render_invariants(frame)
    st.divider()
    render_reconstruction(frame)
    if mode != 'god':
        st.caption('🔒 Изоляция здесь — отладочная фильтрация представлений. '
                   'Приложение хранит полный Snapshot и не является отдельным клиентом '
                   'с криптографически защищённым доступом.')


if __name__ == '__main__':
    main()
