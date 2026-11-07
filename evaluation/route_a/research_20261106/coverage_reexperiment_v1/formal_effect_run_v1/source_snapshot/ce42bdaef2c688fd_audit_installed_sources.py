"""Pin the installed compiler files against this build's wheel RECORD."""
import base64,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def main():
    universe=json.loads((ROOT/'coverage_universe_v1/universe.json').read_text());site=Path('/data1/tzh/pt_nightly_env/lib/python3.11/site-packages')
    record=site/('torch-'+universe['torch']+'.dist-info/RECORD');entries={row[0]:row[1] for row in csv.reader(record.open())};rows=[]
    for path in sorted({r['source_identity'][0] for r in universe['targets']+universe['excluded']}):
        p=Path(path);relative=str(p.relative_to(site));observed='sha256='+base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('=')
        rows.append({'source':path,'wheel_recorded_hash':entries.get(relative),'observed_hash':observed,'matches_current_build_wheel_record':entries.get(relative)==observed})
    assert all(r['matches_current_build_wheel_record'] for r in rows),rows
    (ROOT/'installed_source_audit_v1.json').write_text(json.dumps({'record':str(record),'all_registered_source_files_match_build_record':True,'rows':rows,'not_network_authentication_of_wheel':True},indent=2)+'\n');print('wheel-record source files matched:',len(rows))
if __name__=='__main__':main()
