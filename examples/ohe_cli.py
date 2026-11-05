"""CLI-пример OHE без Streamlit. Запуск: python -m examples.ohe_cli"""
from threshold_voting import Simulation


def main():
    sim = Simulation(
        n=5, t=3, p=101, seed=42,
        options=('Красный', 'Синий', 'Зелёный'),
        choices=(0, 1, 2, 1, 0),
        scenario='honest',
    )
    last = sim.snapshots[-1]
    print('Сценарий:', sim.scenario)
    print('Векторы голосов:', last.votes[1:])
    print('Проверки:', last.checks)
    for subset, counts in last.reconstructed.items():
        print(subset, counts)
    print('Итог:', dict(zip(sim.options, next(iter(last.reconstructed.values())))))


if __name__ == '__main__':
    main()
