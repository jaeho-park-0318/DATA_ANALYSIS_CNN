"""장비별 후속 연구의 별도 계약. 기존 동결 파일은 수정하지 않는다."""
from .common import *
from .pretraining import iter_scoring_batches
from .alarm_evaluation import evaluate
from datetime import datetime,timezone

OUT=ROOT/'outputs/stage7'
PROTOCOL_PATH=ROOT/'config/stage7_specialists.json'
PROTOCOL=json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def utc():return datetime.now(timezone.utc).isoformat()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readjson(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def verify_previous():
    prior=readjson(ROOT/'outputs/stage6/frozen_manifest.json')
    for path,h in prior['sha256'].items():assert digest(ROOT/path)==h,path
    audit=OUT/'prior_artifacts.json'
    if audit.exists():
        for path,h in readjson(audit).items():assert digest(ROOT/path)==h,path


def load(split,tool):
    xs=[];ms=[]
    for x,m in iter_scoring_batches('temporal',split,'B',600,'maximum',tool=tool):xs.append(x);ms.append(m)
    if not xs:return None,None
    x=pd.concat(xs,ignore_index=True);m=pd.concat(ms,ignore_index=True)
    order=np.argsort(m.time.to_numpy(),kind='stable');return x.iloc[order].reset_index(drop=True),m.iloc[order].reset_index(drop=True)


def events_for(tool,split):
    bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet')
    b=bounds[bounds.scheme.eq('temporal')&bounds.split.eq(split)&bounds.Tool.eq(tool)].iloc[0]
    e=pd.read_parquet(TABLES/'events.parquet')
    return e[e.Tool.eq(tool)&e.fault_time.ge(b.start)&e.fault_time.lt(b.end_exclusive)].copy()


def positive_clusters(times):
    t=np.sort(np.unique(np.asarray(times,float)))
    return int(1+np.sum(np.diff(t)>600)) if len(t) else 0


def event_input_counts(m,events):
    alltimes=m.time.to_numpy() if m is not None else np.array([])
    postimes=m.loc[m.audit_label.eq(1),'time'].to_numpy() if m is not None else np.array([])
    available=0;positive=[]
    for e in events[~events.tagged].itertuples():
        lo=e.fault_time-600;hi=e.fault_time-60
        available+=int(np.searchsorted(alltimes,hi,side='right')>np.searchsorted(alltimes,lo))
        if np.searchsorted(postimes,hi,side='right')>np.searchsorted(postimes,lo):positive.append(e.fault_time)
    return dict(recorded_events=len(events),available_events=available,positive_events=len(positive),positive_clusters=positive_clusters(positive))


def grid(scores):
    return np.unique(np.r_[0,.001,.003,.01,.03,.1,.3,.5,.7,.9,1,np.nextafter(1.,2.),np.quantile(scores,[.001,.005,.01,.025,.05,.1,.25,.5,.75,.9,.95,.99,.995,.999,.9995,.9999])])


def alarm_result(p,events,threshold,support):
    s,e,a,l=evaluate(p,events,float(threshold))
    for col in ['tagged','available','detected']:e[col]=e[col].astype(bool)
    s.update(threshold=float(threshold),observed_support_raw=float(support),available_anchors=len(p),alarm_episodes_per_10000_raw=10000*s['alarm_episodes']/support if support else np.nan)
    return s,e,a,l


def choose(curve,budget):
    q=curve[curve.alarm_episodes_per_10000_raw.le(budget)]
    return q.sort_values(['detected_events','alarm_episodes_per_10000_raw','threshold'],ascending=[False,True,False]).iloc[0]
