"""Independent vector/threshold recomputation; does not fit or call runner scores."""
import numpy as np
import realistic_interaction_design as design
import pair_screen_study as io


def vector_scores(comparisons):
    if len(comparisons)!=3:raise ValueError('three CV folds required')
    result=[]
    for pos in (0,1):
        overall=matched=reference=0.;positive=0
        for v in comparisons:
            y=np.asarray(v['observed'],float);baseline=np.asarray(v['baseline_predicted'],float)
            p=v['pairs'][pos];joint=np.asarray(p['predicted'],float);add=np.asarray(p['matched_predicted'],float)
            if not len(y) or any(z.shape!=y.shape for z in (baseline,joint,add)) or not np.isfinite([y,baseline,joint,add]).all():
                raise ValueError('invalid independent prediction vectors')
            br=np.square(y-baseline).sum();jr=np.square(y-joint).sum();ar=np.square(y-add).sum()
            if not np.isfinite(v['reference_mse']) or v['reference_mse']<=0:raise ValueError('invalid reference risk')
            overall+=br-jr;matched+=ar-jr;reference+=len(y)*v['reference_mse'];positive+=bool(ar>jr)
        overall/=reference;matched/=reference;eligible=bool(overall>0 and positive>=2)
        result.append(dict(eligible=eligible,score=float(max(0.,matched) if eligible else 0.)))
    return result


def verify_calibration(audit,manifest_id):
    expected={c['id']:c for c in design.stage_tasks('calibration')};rows=audit.get('rows',[])
    standalone=audit.get('standalone',{})
    if (audit.get('stage')!='calibration' or audit.get('manifest')!=manifest_id or audit.get('planned')!=3600 or
        audit.get('complete')!=3600 or audit.get('missing')!=0 or audit.get('audit_errors') or
        standalone.get('status')!='complete' or standalone.get('checked')!=3600 or
        standalone.get('max_score_difference',float('inf'))>1e-10 or len(rows)!=3600 or
        {r['id'] for r in rows}!=set(expected)):
        raise ValueError('complete independent calibration audit required')
    cutoffs={}
    for row in rows:
        c=expected[row['id']]
        if row['cell']!=c['cell'] or row['benchmark']!=c['benchmark'] or len(row['pairs'])!=2:
            raise ValueError('calibration row identity differs')
        digest=row.get('result_sha256','')
        if len(digest)!=64 or any(ch not in '0123456789abcdef' for ch in digest):raise ValueError('result digest missing')
        for p in row['pairs']:
            if type(p['eligible']) is not bool or type(p['score']) not in (float,int) or not np.isfinite(p['score']) or p['score']<0 or (not p['eligible'] and p['score']!=0):
                raise ValueError('invalid candidate score')
            cutoffs[c['cell']]=max(cutoffs.get(c['cell'],0.),float(p['score']))
    return dict(schema='open.realistic_thresholds.v1',manifest=manifest_id,calibration_audit=io.fingerprint(audit),
        thresholds=dict(sorted(cutoffs.items())),rule='positive_overall_two_positive_folds_strict_fresh_null_max',
        calibration_count=3600,production_accepted=False)
