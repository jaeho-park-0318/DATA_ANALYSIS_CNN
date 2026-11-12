"""표준 라이브러리 로컬 서버. 외부 서버·CDN 요청 없이 Plotly로 표시."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
from functools import lru_cache
import argparse
import mimetypes
import traceback
import pandas as pd
from src.common import *
from src.detail import get_detail,records
from src.query import manifest

@lru_cache(maxsize=20)
def table(name):return pd.read_parquet(TABLES/(name+'.parquet'))

class Handler(BaseHTTPRequestHandler):
    def respond(self,data,status=200,content_type='application/json; charset=utf-8'):
        body=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)

    def do_GET(self):
        route=urlparse(self.path);q={k:v[0] for k,v in parse_qs(route.query).items()}
        try:
            if route.path=='/api/overview':
                data={n:records(table(n)) for n in ['tool_summary','label_status_summary','feature_availability_summary','preprocessing_preservation']}
                data.update(config=CFG,sensors=SENSORS,sensor_names=SENSOR_KO,fault_names=FAULT_KO,policy_names=POLICY_KO,colors=COLORS)
                self.respond(data);return
            if route.path=='/api/events':
                d=table('fault_event_summary');d=d[d.Tool.eq(q['tool'])]
                if q.get('fault') and q['fault']!='any':d=d[d.fault_type.eq(q['fault'])]
                self.respond(records(d));return
            if route.path=='/api/tool':
                tool=q['tool'];m=manifest(tool);d=table('time_segment_summary');d=d[d.Tool.eq(tool)]
                gaps=pd.read_parquet(CACHE/'source'/tool/'gaps.parquet');dups=pd.read_parquet(CACHE/'source'/tool/'duplicates.parquet')
                self.respond({'Tool':tool,'fixed':records(d[d.kind.eq('fixed')]),'continuous':records(d[d.kind.eq('continuous')]),'gaps':records(gaps[gaps.long_gap]),'duplicates':records(dups),'dt_hist':m['dt_hist']});return
            if route.path=='/api/detail':
                self.respond(get_detail(q['tool'],float(q['start']),float(q['end']),q.get('event'),q.get('policy','conservative'),int(q.get('h',600)),q.get('feature','C')));return
            if route.path=='/api/cases':
                path=ROOT/'outputs/case_catalog.json';self.respond(json.loads(path.read_text(encoding='utf-8')) if path.exists() else []);return
            if route.path.startswith('/figures/'):
                base=ROOT/'outputs/figures';path=(base/route.path[len('/figures/'):]).resolve()
            else:
                base=ROOT/'dashboard';path=(base/('index.html' if route.path=='/' else route.path.lstrip('/'))).resolve()
            if not path.is_relative_to(base.resolve()) or not path.is_file():self.respond({'error':'파일 없음'},404);return
            self.respond(path.read_bytes(),content_type=mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        except (ValueError,KeyError,FileNotFoundError) as e:self.respond({'error':str(e)},400)
        except Exception as e:traceback.print_exc();self.respond({'error':str(e)},500)

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    print(f'LOCAL http://127.0.0.1:{a.port}  | V3 진단 전용 · 모델 학습 없음',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
