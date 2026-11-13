"""Independent recomputation of all objective values from saved input coordinates."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import make_objective
ROOT=Path(__file__).resolve().parents[1]

def main():
    reports={}
    for label in ['main','followup']:
        folder=ROOT/'results'/label
        manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['source_sha256']==hashlib.sha256((ROOT/'code'/'benchmark.py').read_bytes()).hexdigest()
        cfg_path=ROOT/'configs'/f'{label}.json'
        assert manifest['config_sha256']==hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        count=0;max_error=0.;warning_count=0
        for path in folder.glob('*.csv'):
            frame=pd.read_csv(path);row=frame.iloc[0]
            objective,_=make_objective(row['function'],int(row.dimension),int(row.trial))
            values=np.array([objective(np.array(json.loads(x))) for x in frame.x])
            err=np.max(np.abs(values-frame.value.to_numpy()));max_error=max(max_error,float(err))
            np.testing.assert_allclose(values,frame.value,rtol=1e-12,atol=1e-10)
            warning_count+=json.loads(path.with_suffix('.json').read_text())['warnings']
            count+=len(frame)
        reports[label]=dict(evaluations_recomputed=count,max_absolute_csv_roundtrip_error=max_error,
                            source_and_config_hash_match=True,fit_warnings_recorded=warning_count)
    target=ROOT/'results'/'summaries'/'objective_audit.json'
    target.write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
