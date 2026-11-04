"""Show local durability, upstream acknowledgement and controlled-run SLA."""

import argparse
from contextlib import closing
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path


def inspect(db_path,sla=False):
    with closing(sqlite3.connect(f'file:{db_path}?mode=ro',uri=True)) as db:
        kinds={kind:count for kind,count in db.execute('SELECT kind,COUNT(*) FROM events GROUP BY kind')}
        pending=db.execute('SELECT COUNT(*) FROM outbox WHERE acknowledged IS NULL').fetchone()[0]
        acked=db.execute('SELECT COUNT(*) FROM outbox WHERE acknowledged IS NOT NULL').fetchone()[0]
        rows=[json.loads(row[0]) for row in db.execute('SELECT payload FROM events ORDER BY sequence')]
    result={'local_event_counts':kinds,'pending_uplink':pending,'acknowledged_uplink':acked,'last_alerts':[r for r in rows if r['kind']=='alert'][-5:]}
    if sla:
        runs=[r for r in rows if r['kind']=='run_start' and r.get('announced_simulator_mode')=='combined']
        if not runs:result['controlled_combined_run_sla']='No combined run has been observed'
        else:
            run=runs[-1];origin=run['origin_unix']
            same=[r for r in rows if r.get('run_id')==run['run_id']]
            critical=[r['created_unix']-origin for r in same if r['kind']=='alert' and r['severity']==2]
            model=[r['created_unix']-origin for r in same if r['kind']=='inference' and r['model_class']==2]
            result['controlled_combined_run_sla']={
                'run_id':run['run_id'],'time_origin':'controlled simulator run start; simulation metadata used only for evaluation',
                'first_local_critical_alert_seconds':min(critical) if critical else None,
                'first_model_critical_seconds':min(model) if model else None,
                'local_critical_within_90_seconds':bool(critical and 0<=min(critical)<90),
                'model_critical_within_90_seconds':bool(model and 0<=min(model)<90)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',default=os.environ.get('EDGE_DB','/state/edge.sqlite'))
    parser.add_argument('--health',action='store_true');parser.add_argument('--sla-latest-combined',action='store_true')
    args=parser.parse_args()
    health_path=Path(os.environ.get('HEALTH_PATH','/state/health.json'))
    if args.health:
        try:
            value=json.loads(health_path.read_text())
            return 0 if time.time()-value['updated_unix']<15 and value['model_ready'] and value['mqtt_connected'] else 1
        except (OSError,ValueError,KeyError):return 1
    result=inspect(args.db,args.sla_latest_combined)
    if health_path.exists():result['service']=json.loads(health_path.read_text())
    url=os.environ.get('OPERATIONS_URL','http://operations:8080/events').rsplit('/',1)[0]+'/stats'
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url,timeout=2) as response:result['operations_centre']=json.loads(response.read())
    except Exception as error:result['operations_centre']={'available':False,'reason':str(error)}
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
