"""Streamlit-интерфейс OHE-порогового голосования с тремя malicious-атаками.

Запуск из корня репозитория:
    python -m streamlit run voting_ui/app.py
Криптографические вычисления выполняет threshold_voting/src.core, не UI.
"""
from __future__ import annotations
from itertools import combinations
import hashlib
import streamlit as st

from ui_layer.contracts import Trace, parse_trace, trace_to_json
from ui_layer.graph import build_dot
from ui_layer.observer import reconstruct_public, alternative_curves, horner
from ui_layer.projections import Frame, project, safe_description
from ui_layer.source import load_backend

st.set_page_config(page_title='ConfidentialVoting · OHE', page_icon='🔐', layout='wide')
st.markdown('''<style>
.block-container {padding-top:1.35rem;max-width:1600px}
[data-testid="stMetricValue"] {font-size:1.65rem}
</style>''', unsafe_allow_html=True)


def vec_text(vector: tuple[int,...] | None, options: tuple[str,...]) -> str:
    if vector is None:
        return '?'
    return '(' + ', '.join(map(str,vector)) + ')'


def polynomial_text(poly: tuple[int,...], option: str, j: int) -> str:
    terms = [str(poly[0])]
    for d,c in enumerate(poly[1:],1):
        terms.append(f'{c}·x' if d==1 else f'{c}·x^{d}')
    return f'f_{j},{option}(x) = ' + ' + '.join(terms)


def load_trace() -> tuple[Trace | None,str]:
    with st.sidebar:
        st.title('⚙️ Конфигурация')
        mode=st.radio('Данные', ['Реальная OHE-симуляция','Загрузить JSON'],
                      help='Для JSON предполагается векторный контракт OHE v2')
        if mode=='Загрузить JSON':
            uploaded=st.file_uploader('trace_ohe.json',type=['json'])
            if uploaded is None:
                st.info('Загрузите трассу OHE; файл можно выгрузить из режима симуляции.')
                return None,'no-json'
            payload=uploaded.getvalue()
            return parse_trace(payload),'json:'+hashlib.sha256(payload).hexdigest()

        n=int(st.number_input('Участников n',min_value=2,max_value=10,value=5))
        t=int(st.number_input('Порог t',min_value=2,max_value=n,value=min(3,n)))
        p=int(st.number_input('Поле Fₚ (простое p > n)',min_value=n+1,value=101))
        seed=int(st.number_input('Seed',value=42,step=1))
        options_text=st.text_input('Варианты, через запятую',value='Альфа, Бета, Гамма')
        options=tuple(x.strip() for x in options_text.split(',') if x.strip())
        if len(options)<2 or len(options)>10 or len(set(options))!=len(options):
            st.error('Укажи от 2 до 10 разных названий вариантов.')
            return None,'invalid-options'
        st.subheader('Выбор каждого участника')
        selected_choices=[]
        for i in range(1,n+1):
            initial=(i-1)%len(options)
            name=st.selectbox(f'Участник {i}',options,index=initial,key=f'choice_{i}',
                              help='Каждый участник выбирает ровно один вариант; голос становится OHE-вектором.')
            selected_choices.append(options.index(name))
        choices=tuple(selected_choices)
        scenario_labels={
            'honest':'✅ Honest',
            'invalid_vote':'🔴 1. Invalid OHE vote (3 вместо 1)',
            'wrong_private_share':'🔴 2. Incorrect private share fᵢ(j)',
            'wrong_public_share':'🔴 3. Incorrect published share sⱼ',
        }
        scenario=st.selectbox('Сценарий',list(scenario_labels),
                              format_func=lambda name:scenario_labels[name])
        attacker=2 if n>=2 else 1
        delta=7
        receiver=None
        invalid_secret=3
        attack_option=None
        if scenario!='honest':
            attacker=int(st.selectbox('Злоумышленник',list(range(1,n+1)),index=min(1,n-1)))
            if scenario=='invalid_vote':
                invalid_secret=int(st.number_input('Некорректное значение выбранной координаты',
                                    min_value=2,max_value=p-1,value=min(3,p-1)))
                st.caption('Вектор (0,1,0) → (0,3,0); все координатные многочлены остаются согласованными.')
            else:
                attack_option=int(st.selectbox('Искажаемая координата/вариант',list(range(len(options))),
                                               format_func=lambda k:options[k]))
                delta=int(st.number_input('Прибавка Δ (mod p)',min_value=1,max_value=p-1,value=min(7,p-1)))
                if scenario=='wrong_private_share':
                    receiver=int(st.selectbox('Кому передать неверную fᵢ(j)',
                                              [i for i in range(1,n+1) if i!=attacker]))
                    st.caption('Меняется одна координата одной приватной доли до локального суммирования.')
                else:
                    st.caption('Меняется одна координата одного публичного вектора sⱼ.')

        st.divider()
        st.subheader('Этап 3 · Консенсус и исключение')
        enable_consensus=st.checkbox('Включить голосование об исключении malicious-агентов',value=True)
        detector_mode='none'
        manual_accusations={}
        max_consensus_rounds=10
        if enable_consensus:
            detector_mode=st.selectbox('Откуда брать подозрения участников',
                ('demo_oracle', 'manual', 'none', 'external'),
                format_func=lambda key: {
                    'demo_oracle':'⚠️ Демо: всем известен заданный атакующий (не криптография)',
                    'manual':'Ручные MaliciousReport каждого участника',
                    'none':'Нет обвинений / детектор пока не подключён',
                    'external':'🔧 Мой детектор (threshold_voting/detector_adapter.py)'
                }[key])
            if detector_mode=='external':
                st.info('Для этого режима реализуй detect(observer, suspect, evidence) '
                        'в threshold_voting/detector_adapter.py.')
            if detector_mode=='demo_oracle':
                st.warning('Учебный оракул знает malicious-id из параметров запуска. '
                           'В реальном протоколе это недоступно; подключи свой детектор.')
            elif detector_mode=='manual':
                st.caption('Отметь, кого каждый участник считает malicious. '
                           'MaliciousReport публикуются публично, порог строго > 50%.')
                for observer in range(1,n+1):
                    candidates=[suspect for suspect in range(1,n+1) if suspect!=observer]
                    manual_accusations[observer]=tuple(st.multiselect(
                        f'P{observer} подозревает', candidates, key=f'accusations_{n}_{observer}'))
            max_consensus_rounds=int(st.number_input('Максимум раундов',
                                      min_value=1,max_value=30,value=10))

        accusations_key=tuple(sorted((i, tuple(v)) for i,v in manual_accusations.items()))
        params=(n,t,p,seed,choices,options,scenario,attacker,delta,receiver,
                invalid_secret,attack_option,enable_consensus,detector_mode,
                accusations_key,max_consensus_rounds)
        cache_key=repr(params)
        if st.button('Пересчитать',type='primary',use_container_width=True) or st.session_state.get('trace_config')!=cache_key:
            try:
                trace=load_backend(n,t,p,seed,choices,options,scenario,attacker,delta,receiver,
                                   invalid_secret,attack_option,
                                   enable_consensus=enable_consensus,
                                   detector_mode=detector_mode,
                                   manual_accusations=manual_accusations,
                                   max_consensus_rounds=max_consensus_rounds)
            except (ValueError,TypeError,ImportError,AttributeError,ZeroDivisionError,NotImplementedError) as exc:
                st.error(f'Ошибка расчёта: {exc}')
                st.session_state.pop('trace_data',None)
                st.session_state.pop('trace_config',None)
                return None,'error'
            st.session_state['trace_data']=trace
            st.session_state['trace_config']=cache_key
        return st.session_state.get('trace_data'),'real:'+cache_key


def render_attack(frame:Frame) -> None:
    info=frame.attack_info
    if not info: return
    kind=info['type']
    who=info['attacker']
    if kind=='invalid_vote':
        st.error(f'🔴 Атака 1 · P{who} использовал недопустимый one-hot бюллетень: '
                 f'{tuple(info["actual_vector"])} вместо {tuple(info["original_vector"])}.')
        c1,c2=st.columns(2)
        c1.metric('Ожидаемый результат',vec_text(tuple(info['expected_counts']),frame.options))
        c2.metric('Фактическая сумма секретов',vec_text(tuple(info['actual_counts']),frame.options))
        st.caption('Все реконструкции совпадают, но OHE нарушен. Статус OHE доступен '
                   'только God View; проверка допустимости без доказательства ZKP не обеспечена.')
    elif kind=='wrong_private_share':
        st.error(f'🔴 Атака 2 · P{who} отправил P{info["receiver"]} '
                 f'неправильную координату «{frame.options[info["option"]]}» fᵢ(j): '
                 f'{tuple(info["sent_vector"])} вместо {tuple(info["expected_vector"])}.')
        st.caption('Локальная сумма получателя соответствует ПОЛУЧЕННЫМ данным, '
                   'но исходные многочлены и опубликованные доли больше не согласованы.')
    elif kind=='wrong_public_share':
        st.error(f'🔴 Атака 3 · P{who} опубликовал ошибочную sⱼ в координате '
                 f'«{frame.options[info["option"]]}»: '
                 f'{tuple(info["published_vector"])} вместо {tuple(info["expected_vector"])}.')
        st.caption('Приватные сообщения и локальные суммы корректны; искажена только публикация.')


def render_consensus(frame:Frame) -> None:
    """Публичные заявления участников и результат циклического исключения."""
    if frame.round_number == 0:
        return
    st.subheader(f'🗳️ Этап 3 · Раунд {frame.round_number}: MaliciousReport')
    active_text=', '.join(f'P{i}' for i in frame.active_participants)
    excluded_text=', '.join(f'P{i}' for i in frame.excluded_participants) or 'нет'
    a,b,c=st.columns(3)
    a.metric('Остались активны',len(frame.active_participants))
    b.metric('Всего исключено',len(frame.excluded_participants))
    c.metric('Текущий порог Шамира t',frame.t)
    st.caption(f'Активные: {active_text or "нет"}. Исключённые: {excluded_text}. '
               'Решение о malicious принимается строго >50% активных в данном раунде.')
    st.caption(f'Состояние VotingSession: `{frame.session_phase}`. '
               'ID участников не перенумеровываются после исключений.')
    if frame.membership_events:
        with st.expander('История регистрации и исключения участников'):
            st.dataframe([{'Событие': event, 'Участник': f'P{idx}',
                           'Раунд': '-' if rid is None else rid}
                          for event, idx, rid in frame.membership_events],
                         hide_index=True, use_container_width=True)
    round_messages=[v for v in frame.reports if v.round_id == frame.round_number]
    electorate = frame.round_electorate or frame.active_participants
    if round_messages:
        observers=sorted({v.reporter for v in round_messages})
        min_votes=len(electorate)//2+1
        st.caption(f'Строгое большинство: минимум {min_votes} из {len(electorate)} '
                   'участников раунда; собственный голос подозреваемого не считается обвинением.')
        with st.expander('Публичные MaliciousReport (broadcast)',expanded=frame.phase in ('report','tally')):
            st.dataframe([{'Раунд':v.round_id,'Кто голосует':f'P{v.reporter}',
                           'Подозреваемый':f'P{v.suspect}',
                           'MaliciousReport':'malicious' if v.is_malicious else 'не обвиняет'}
                          for v in round_messages],hide_index=True,use_container_width=True)
    if frame.tallies:
        st.dataframe([{'Агент':f'P{i}', 'Обвинений':votes,
                       'Строгое большинство': 'Да' if votes >= (len(electorate)//2+1)
                       else 'Нет'} for i,votes in frame.tallies],
                     hide_index=True,use_container_width=True)
    if frame.new_exclusions:
        st.error('Исключены решением большинства: '+', '.join(f'P{i}' for i in frame.new_exclusions))
    if frame.phase=='restart':
        st.warning(f'Осталось меньше прежнего t: запущено новое распределение долей с t={frame.t}.')
    if frame.phase=='reaggregate':
        st.success('Новые доли пересчитаны и опубликованы без исключённых участников. '
                   'Использовано вычитание по модулю p, НЕ XOR.')
    if frame.consensus_status=='finalized' and frame.final_result is not None:
        st.success('✅ Итог утверждён после проверки голосов и согласованности: '
                   +vec_text(frame.final_result,frame.options))
        maxi=max(frame.final_result)
        winners=[frame.options[k] for k,value in enumerate(frame.final_result) if value==maxi]
        st.markdown('**Победитель:** '+', '.join(winners)+(' (ничья)' if len(winners)>1 else ''))
    elif frame.consensus_status=='restarted':
        st.warning(f'Порог снизился: свежий протокол Шамира с t={frame.t} и новыми многочленами.')
    elif frame.consensus_status=='terminated_too_few':
        st.error(f'Осталось {len(frame.active_participants)} участников (<2): '
                 'протокол завершён принудительно, результат не объявляется.')
    elif frame.consensus_status=='unresolved':
        st.error('Нет новых исключений, но согласованный допустимый итог не подтверждён.')
    elif frame.consensus_status=='max_rounds':
        st.warning('Достигнут максимум раундов: итог не утверждён.')
    st.caption('Важно: простое большинство не доказывает правильность детектора; '
               'ошибочные обвинения могут исключить честного участника. '
               'Утверждение результата проверяет согласованность и сумму счётчиков, '
               'но не заменяет ZKP допустимости каждого OHE-бюллетеня.')


def render_cards(frame:Frame) -> None:
    st.subheader('Участники')
    for card in frame.cards:
        title=f'{"⛔" if card.idx in frame.excluded_participants else "🔓" if card.can_see_private else "🔒"} P{card.idx} · '+('исключён' if card.idx in frame.excluded_participants else 'опубликовано' if card.published else 'пока не опубликовано')
        with st.expander(title,expanded=card.idx==frame.active):
            a,b=st.columns(2)
            a.metric('OHE vᵢ',vec_text(card.vote,frame.options))
            b.metric('Вектор sᵢ',vec_text(card.share,frame.options))
            if card.vote is not None:
                st.caption('Координаты: '+', '.join(f'{frame.options[k]}={v}' for k,v in enumerate(card.vote)))
            if card.polys is not None:
                for k,poly in enumerate(card.polys):
                    st.code(polynomial_text(poly,frame.options[k],card.idx),language=None)
            else:
                st.caption('Чужие многочлены скрыты.')
            if card.received:
                st.dataframe([{'От':i,'Полученная fᵢ(j)':vec_text(vector,frame.options)}
                              for i,vector in card.received],hide_index=True,use_container_width=True)
            elif card.can_see_private:
                st.caption('Входящие доли пока не получены.')


def render_checks(frame:Frame) -> None:
    st.subheader('Диагностики')
    st.caption('I1 — предупреждение о совпадении, OHE — допустимость бюллетеня, '
               'I6 требует отдельного доказательства и теста неразличимости. '
               'В ограниченных видах приватные диагностики скрыты.')
    cols=st.columns(8)
    status={'ok':'🟢','warning':'🟡','fail':'🔴','pending':'⏳','hidden':'🔒','unknown':'⚪'}
    for c,name in zip(cols,['I1','I2','I3','I4','I5','I6','I7','OHE']):
        value=frame.checks.get(name,'unknown')
        c.markdown(f'**{name}** {status.get(value,"⚪")}')
        c.caption(value)


def render_reconstruction(frame:Frame) -> None:
    st.subheader('Реконструкция Лагранжем · покоординатно')
    if frame.consensus_status == 'terminated_too_few':
        st.error('Реконструкция запрещена: осталось меньше двух участников.')
        return
    if frame.phase in ('exclude','restart','redeal'):
        st.warning('Участники только что исключены: пересчёт долей ещё не выполнен.')
        return
    if frame.consensus_status in ('terminated_too_few','max_rounds','restarted'):
        st.info('Финальный результат не утверждён; ниже — только текущие публикации.')
    published={d.sender:d.value for d in frame.board_deals}
    available=sorted(published)
    if not available:
        st.info('Публичных векторов sⱼ пока нет.')
        return
    chosen=st.multiselect('Выбрать опубликованные доли',available,
                          default=available[:min(len(available),frame.t)],
                          key=f'reconstruct_{frame.step}')
    selected=sorted(chosen)
    possible={reconstruct_public({j:published[j] for j in subset},frame.t,frame.p)
              for subset in combinations(available,frame.t)} if len(available)>=frame.t else set()
    if len(selected)==frame.t:
        counts=reconstruct_public({j:published[j] for j in selected},frame.t,frame.p)
        if len(possible)>1:
            st.error(f'⚠️ По выбранным точкам: {counts}, но другие t-наборы дают иной вектор!')
        else:
            st.success(f'Восстановленный вектор счётчиков: {counts}')
            st.bar_chart([{'Вариант':frame.options[k], 'Голосов':counts[k]} for k in range(len(counts))], x='Вариант', y='Голосов')
            if sum(counts)!=len(frame.active_participants or tuple(range(1,frame.n+1))):
                st.warning('Сумма счётчиков не равна количеству активных. Это признак аномалии, '
                           'но выполнение условия ΣS=n не доказывает корректность каждого OHE.')
    elif len(selected)<frame.t:
        st.warning(f'Выбрано {len(selected)} < t={frame.t}; нельзя восстановить вектор.')
        curves=alternative_curves({j:published[j][0] for j in selected},frame.t,frame.p)
        st.caption(f'Для координаты «{frame.options[0]}» две совместимые кривые '
                   f'имеют свободные члены 0 и 1: {curves[0]}, {curves[1]}.')
        if len(available)>=frame.t:
            st.caption('Другие опубликованные доли уже могут определять результат.')
    else:
        st.warning('Для интерполяции выбери ровно t долей.')
    if len(possible)>1:
        st.error(f'Проверка согласованности: {len(possible)} различных векторов '
                 'получены из t-наборов. Нет единого итога.')
    if len(available)>=frame.t and (frame.phase in ('reconstruct','reaggregate','finalize')):
        with st.expander(f'Все C({frame.n},{frame.t}) комбинаций',expanded=True):
            rows=[]
            for subset in combinations(available,frame.t):
                counts=reconstruct_public({j:published[j] for j in subset},frame.t,frame.p)
                rows.append({'Доли':str(subset),**{frame.options[k]:counts[k] for k in range(len(counts))}})
            st.dataframe(rows,hide_index=True,use_container_width=True)
            if len(possible)==1:
                st.success('Все наборы восстановили одинаковый вектор счётчиков.')


def main() -> None:
    st.title('🔐 ConfidentialVoting · OHE Protocol Debugger')
    st.caption('OHE → Шамир → Core Board (f_i(j), F(j), MaliciousReport) → '
               'строгое большинство → исключение → пересчёт / перезапуск.')
    try:
        trace,key=load_trace()
    except (ValueError,TypeError,KeyError) as exc:
        st.error(f'Ошибка ввода: {exc}')
        st.stop()
    if trace is None: st.stop()
    if st.session_state.get('key')!=key:
        st.session_state['step']=1
        st.session_state['key']=key
    st.session_state['step']=max(1,min(st.session_state['step'],len(trace.snapshots)))
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Участники',trace.n)
    c2.metric('Начальный порог',trace.t)
    c3.metric('Варианты',len(trace.options))
    c4.metric('p',trace.p)
    st.write('**Варианты:** '+', '.join(f'{k}: {v}' for k,v in enumerate(trace.options)))
    st.write(f'**Сценарий:** `{trace.scenario}`')
    mode=st.radio('View as',['god','participant','coalition'],horizontal=True)
    selection=()
    if mode=='participant':
        selection=(int(st.selectbox('Чей взгляд',list(range(1,trace.n+1)))),)
    elif mode=='coalition':
        picked=st.multiselect('Коалиция',list(range(1,trace.n+1)),default=[1,2])
        if not picked: st.stop()
        selection=tuple(sorted(picked))
        if len(selection)>=trace.t:
            st.warning('Размер коалиции ≥ t: может восстановить чужие индивидуальные голоса.')
    a,b,c,_=st.columns([1,1,1,5])
    if a.button('◀ Назад',disabled=st.session_state['step']==1):st.session_state['step']-=1
    if b.button('Вперёд ▶',disabled=st.session_state['step']==len(trace.snapshots)):st.session_state['step']+=1
    if c.button('↺ Начало'):st.session_state['step']=1
    st.slider('Шаг',1,len(trace.snapshots),key='step')
    snap=trace.snapshots[st.session_state['step']-1]
    frame=project(trace,snap,mode,selection)
    st.info(f'Шаг {frame.step}/{frame.total_steps} · t={frame.t} · {frame.event}')
    if mode=='god':render_attack(frame)
    render_consensus(frame)
    left,right=st.columns([1.7,1.0])
    with left:
        st.subheader('Граф передачи ВЕКТОРОВ долей')
        st.graphviz_chart(build_dot(frame),use_container_width=True)
        st.caption('Каждая метка вида [a,b,c] — доли по вариантам '+', '.join(frame.options)+'.')
    with right:render_cards(frame)
    log_col,board_col=st.columns(2)
    with log_col:
        st.subheader('Лог этапов')
        for s in trace.snapshots[:frame.step]:
            st.text(f'{s.step:02d}. '+safe_description(s.phase,s.active,trace.n,trace.t))
    with board_col:
        st.subheader('Core BulletinBoard · журнал')
        if frame.board_events:
            st.dataframe([{'ID':id_, 'Тип':kind, 'От P':sender,
                           'К P':receiver if receiver is not None else 'всем',
                           'Раунд':rnd}
                          for id_,kind,sender,receiver,rnd in frame.board_events],
                         hide_index=True,use_container_width=True)
            st.caption('PRIVATE_SHARE: на доске пока открытый вектор; '
                       'participant видит только адресованные ему значения. '
                       'С будущим шифрованием все увидят ciphertext.')
        if frame.board_deals:
            st.markdown('**Последние опубликованные F(j):**')
            st.dataframe([{'Участник':d.sender,'F(j)':vec_text(d.value,frame.options)}
                          for d in frame.board_deals],hide_index=True,use_container_width=True)
        elif not frame.board_events:
            st.caption('Записей пока нет.')
    st.divider()
    render_checks(frame)
    st.divider()
    render_reconstruction(frame)
    if mode=='god':
        st.download_button('⬇️ Скачать полную трассу (с приватными данными!)',
                           data=trace_to_json(trace),file_name='ohe_trace_god.json',
                           mime='application/json')
    else:
        st.caption('🔒 Ограниченный режим — визуальная проекция единого God Snapshot; '
                   'это не изолированный защищённый клиент.')

if __name__=='__main__':
    main()
