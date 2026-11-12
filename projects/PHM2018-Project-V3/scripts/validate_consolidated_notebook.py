"""종합 Notebook의 저장 실행·내장 그림·본문 링크를 검증한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ast,re,hashlib
import nbformat as nb
from src.common import ROOT,dump_json
from src.specialist_experiment import verify_previous


def main():
    path=ROOT/'notebooks/00_PHM2018_종합_수행결과.ipynb'
    book=nb.read(path,as_version=4);nb.validate(book)
    markdown='\n'.join(c.source for c in book.cells if c.cell_type=='markdown')
    anchors=set(re.findall(r'id="([^"]+)"',markdown));refs=re.findall(r'\]\(#([^)]+)\)',markdown)
    assert set(refs)<=anchors,set(refs)-anchors
    figures=[c for c in book.cells if c.get('attachments')]
    assert len(figures)==123
    assert all('attachment:figure.png' in c.source and 'image/png' in c.attachments['figure.png'] for c in figures)
    codes=[c for c in book.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in codes)
    assert not [o for c in codes for o in c.outputs if o.output_type=='error']
    for c in codes:ast.parse(c.source)
    verify_previous()
    result=dict(schema_valid=True,internal_links=len(refs),anchors=len(anchors),figure_attachments=len(figures),code_cells=len(codes),all_code_executed=True,errors=0,archived_reports=markdown.count('<details>'),notebook_MB=round(path.stat().st_size/1000000,2),notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prior_artifacts_unchanged=True)
    dump_json(ROOT/'outputs/consolidated/structure_validation.json',result);print(result)


if __name__=='__main__':main()
