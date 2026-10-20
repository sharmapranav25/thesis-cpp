import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from feature_bank.iter2 import balanced as B

if __name__ == '__main__':
    ticket = json.loads(Path(r'E:\pranav\Final sem\sourcefree_synth\runs\iter2_baseline_15_20261003_01\steering_failed_tickets.json').read_text())[0]['ticket']
    tickets = []
    for weight, width in [(1, 9), (3, 9), (5, 9), (1, 5), (3, 5), (5, 5)]:
        item = copy.deepcopy(ticket)
        item.update(weight=weight, width=width, revision=7000, max_attempts=40)
        tickets.append(item)
    with ProcessPoolExecutor(6) as pool:
        for ticket, row in zip(tickets, pool.map(B.render_ticket, tickets)):
            print(json.dumps(dict(weight=ticket['weight'], width=ticket['width'], failed=row['failed'], rejections=row['rejections'], quality=row.get('label', {}).get('quality'))), flush=True)
