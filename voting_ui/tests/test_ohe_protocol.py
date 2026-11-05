"""Интеграционные проверки OHE-ядра и трёх атак; запускать из корня репо."""
import sys
from pathlib import Path
from itertools import combinations
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'voting_ui'))
from src.core.vector import one_hot, is_one_hot, VectorShare, reconstruct_vector
from threshold_voting.simulation import Simulation
from ui_layer.source import load_backend
from ui_layer.projections import project
from ui_layer.observer import reconstruct_public
from ui_layer.contracts import trace_to_json, parse_trace

OPTIONS=('A','B','C','D')
CHOICES=(0,1,2,1,3)
EXPECTED=(1,2,1,1)


def make(scenario='honest', **kwargs):
    return Simulation(5,3,101,seed=17,choices=CHOICES,options=OPTIONS,
                      scenario=scenario,attacker=2,**kwargs)


def test_ohe_encoding_and_invalid_vector():
    assert one_hot(2,4)==(0,0,1,0)
    assert is_one_hot((0,1,0,0))
    assert not is_one_hot((0,3,0,0))
    assert not is_one_hot((1,1,0,0))
    assert not is_one_hot((0,0,0,0))
    with pytest.raises(ValueError): one_hot(4,4)


def test_honest_and_repeatable():
    a,b=make(),make()
    assert a.snapshots==b.snapshots
    assert len(a.snapshots)==13
    final=a.snapshots[-1]
    assert len(final.reconstructed)==10
    assert set(final.reconstructed.values())=={EXPECTED}
    assert all(final.aggregate_F[k][0]==EXPECTED[k] for k in range(4))
    assert final.checks['I2']=='ok'
    assert final.checks['I4']=='ok'
    assert final.checks['I5']=='ok'
    assert final.checks['I7']=='ok'
    assert final.checks['OHE']=='ok'


def test_public_vectors_are_equal_to_polynomial_aggregate():
    final=make().snapshots[-1]
    assert all(len(final.shares[j])==4 for j in range(1,6))
    assert all(len(final.received[j])==4 for j in range(1,6))
    assert all(final.shares[j]==tuple(d.value for d in final.all_deals
                                     if d.receiver==-1 and d.sender==j)[0]
               for j in range(1,6))


@pytest.mark.parametrize('variants,choices,expected',[
    (('A','B'),(0,1,1,0,1),(2,3)),
    (('A','B','C'),(2,0,1,2,2),(1,1,3)),
    (('A','B','C','D','E'),(0,1,2,3,4),(1,1,1,1,1)),
])
def test_different_option_counts(variants,choices,expected):
    sim=Simulation(5,3,101,23,choices,variants)
    assert set(sim.snapshots[-1].reconstructed.values())=={expected}


def test_wrong_vote_is_consistent_but_invalid():
    final=make('invalid_vote',invalid_secret=3).snapshots[-1]
    assert set(final.reconstructed.values())=={(1,4,1,1)}
    assert final.checks['I5']=='ok'
    assert final.checks['I7']=='ok'
    assert final.checks['OHE']=='fail'
    assert final.votes[2]==(0,3,0,0)


@pytest.mark.parametrize('kind',['wrong_private_share','wrong_public_share'])
def test_tamper_yields_inconsistent_reconstructions(kind):
    final=make(kind,attack_delta=7,attack_option=2,target_receiver=4).snapshots[-1]
    assert len(set(final.reconstructed.values()))>1
    assert final.checks['I5']=='fail'
    assert final.checks['I7']=='fail'
    assert final.checks['OHE']=='ok'
    assert final.checks['I4']=='ok'
    assert final.votes[2]==(0,1,0,0)


def test_private_and_public_attack_are_same_if_same_recipient():
    a=make('wrong_private_share',target_receiver=4,attack_option=1).snapshots[-1]
    b=Simulation(5,3,101,17,CHOICES,OPTIONS,'wrong_public_share',attacker=4,
                 attack_delta=7,attack_option=1).snapshots[-1]
    assert [d.value for d in a.all_deals if d.receiver==-1]==[d.value for d in b.all_deals if d.receiver==-1]
    assert a.received[4][2]!=b.received[4][2]


def test_ui_adapter_and_view_isolation():
    trace=load_backend(5,3,101,17,CHOICES,OPTIONS,'invalid_vote',attacker=2)
    snap=trace.snapshots[-1]
    assert snap.votes[2]==(0,3,0,0)
    single=project(trace,snap,'participant',(1,))
    assert single.cards[1].vote is None
    assert single.cards[0].vote==(1,0,0,0)
    assert single.attack_info is None
    assert single.checks['OHE']=='hidden'
    assert all(d.receiver==-1 or d.sender==1 or d.receiver==1 for d in single.visible_deals)
    coalition=project(trace,snap,'coalition',(1,3))
    assert coalition.cards[1].vote is None
    assert coalition.cards[2].vote==(0,0,1,0)
    god=project(trace,snap,'god')
    assert god.attack_info['type']=='invalid_vote'
    assert god.checks['OHE']=='fail'


def test_json_roundtrip_and_public_interpolation():
    trace=load_backend(5,3,101,17,CHOICES,OPTIONS)
    decoded=parse_trace(trace_to_json(trace))
    assert decoded==trace
    last=decoded.snapshots[-1]
    published={d.sender:d.value for d in last.all_deals if d.receiver==-1}
    assert all(reconstruct_public({i:published[i] for i in subset},3,101)==EXPECTED
               for subset in combinations(range(1,6),3))


def test_invalid_parameters():
    with pytest.raises(ValueError): Simulation(5,3,4,17,CHOICES,OPTIONS)
    with pytest.raises(ValueError): Simulation(5,3,101,17,(0,1),OPTIONS)
    with pytest.raises(ValueError): Simulation(5,3,101,17,CHOICES,('A',))
    with pytest.raises(ValueError): Simulation(5,3,101,17,CHOICES,OPTIONS,'invalid_vote',invalid_secret=1)
