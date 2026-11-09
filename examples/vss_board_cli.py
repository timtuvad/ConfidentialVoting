"""Пример: python -m examples.vss_board_cli (из корня проекта)."""
from threshold_voting.vss_protocol import VSSVotingProtocol


def main():
    for scenario in ('honest', 'wrong_private_share', 'wrong_public_share',
                     'false_complaint_changed', 'invalid_vote'):
        protocol = VSSVotingProtocol((0, 1, 2, 1, 0), 3, t=3, p=23,
                                      scenario=scenario, attacker=2, receiver=4)
        result = protocol.run()
        print(f'{scenario:24} status={result.status:<18} '
              f'active={result.active_ids} excluded={result.excluded_ids} '
              f'counts={result.counts}')
        for round_info in result.rounds:
            print(f'  round {round_info.round_id}, t={round_info.threshold}, '
                  f'{round_info.phase}, verdicts={len(round_info.verdicts)}')


if __name__ == '__main__':
    main()
