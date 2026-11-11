"""标准库调度的Adam启动诊断；不加载模型/数据/权重，不修改依赖或正则实现。"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

WORKER = r'''
import sys
mode, project = sys.argv[1:3]
if mode in ('project','preload_project'):
    if sys.flags.isolated:
        sys.path.insert(0, project)
    else:
        sys.path[0:1] = [project]
import hashlib
import importlib.metadata
import json
import re
import sys
import traceback
from pathlib import Path

phase = '初始状态'

def emit(**value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)

def module_info(name):
    module = sys.modules.get(name)
    path = getattr(module, '__file__', None)
    if not path:
        return dict(loaded=module is not None, file=None)
    p = Path(path).resolve()
    try:
        fingerprint = hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        fingerprint = None
    return dict(loaded=True, file=str(p), tail='/'.join(p.parts[-3:]), sha256=fingerprint,
                within_prefix=Path(sys.prefix).resolve() in p.parents)

def regex_state():
    parser = sys.modules.get('sre_parse') or sys.modules.get('re._parser')
    letters = getattr(parser, 'ASCIILETTERS', None)
    compile_function = re.compile
    code = getattr(compile_function, '__code__', None)
    return dict(re_module=module_info('re'), parser_module=module_info(getattr(parser, '__name__', 'sre_parse')),
        compile_module=getattr(compile_function, '__module__', None),
        compile_code_sha256=hashlib.sha256(code.co_code).hexdigest() if code else None,
        ascii_letters_count=len(letters) if letters is not None else None,
        slash_in_ascii_letters=('/' in letters) if letters is not None else None,
        escaped_probe=re.escape('固定目录/a-b.c'))

versions={}
for package in ('torch','numpy','scipy','sympy'):
    try:
        versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package]=None
emit(stage='导入前', mode=mode, python=sys.version, executable=sys.executable,
     isolated=sys.flags.isolated, versions=versions, regex=regex_state())
try:
    phase='导入torch'
    import torch
    torch.set_num_threads(1)
    emit(stage='torch导入成功', torch_version=str(torch.__version__),
         compiled_cuda_version=torch.version.cuda, regex=regex_state())
    if mode=='preload_project':
        phase='先于项目预加载torch._dynamo'
        import torch._dynamo
        emit(stage='预加载完成', module=module_info('torch._dynamo.trace_rules'))
    if mode in ('project','preload_project'):
        phase='仅导入项目模块定义'
        import phase6_synthetic_experiment
        import NeuroFDAN
        emit(stage='项目模块导入成功', regex=regex_state())
    phase='创建两元素CPU参数张量的Adam'
    parameter=torch.nn.Parameter(torch.tensor([1., -1.], device='cpu'))
    optimizer=torch.optim.Adam([parameter], lr=.001, weight_decay=0.)
    emit(stage='Adam初始化成功', dynamo=module_info('torch._dynamo.trace_rules'), regex=regex_state())
    phase='一次CPU微型优化器检查'
    before=parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    parameter.square().sum().backward()
    optimizer.step()
    if not torch.isfinite(parameter).all() or torch.equal(parameter.detach(), before):
        raise RuntimeError('微型参数未正常更新或出现非有限值')
    emit(stage='检查通过', device=str(parameter.device), dummy_cpu_optimizer_step=True,
         model_loaded=False, real_or_synthetic_training=False,
         regex=regex_state(), parameter_update_norm=float((parameter.detach()-before).norm()))
except Exception as error:
    pattern=getattr(error, 'pattern', None)
    position=getattr(error, 'pos', None)
    pattern_data={}
    if isinstance(pattern, (str,bytes)):
        raw=pattern.encode('utf-8') if isinstance(pattern,str) else pattern
        pattern_data=dict(length=len(pattern), sha256=hashlib.sha256(raw).hexdigest(), position=position)
        if isinstance(position,int):
            segment=pattern[position:position+2]
            pattern_data['at_error_character_codes']=[ord(v) for v in segment] if isinstance(segment,str) else list(segment)
    try:
        regex=regex_state()
    except Exception as secondary:
        regex=dict(state_read_error=type(secondary).__name__)
    emit(stage='检查失败', failed_phase=phase, error_type=type(error).__name__,
         error_message=str(error), pattern=pattern_data, regex=regex)
    traceback.print_exc()
    raise SystemExit(1)
'''


def as_text(value) -> str:
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''


def main() -> None:
    parser=argparse.ArgumentParser(description='全新进程比较Adam延迟导入；不安装依赖、不运行NeuroFDAN',add_help=False)
    parser.add_argument('-h','--help',action='help',help='显示中文帮助')
    parser.add_argument('--output_dir',type=Path,required=True,help='独立新诊断目录')
    parser.add_argument('--project_dir',type=Path,default=PROJECT,help='待诊断的原Phase6工具根目录，不执行其main函数')
    parser.add_argument('--repeats',type=int,choices=[1,2,3],default=2,help='每种模式独立进程检查次数')
    parser.add_argument('--timeout',type=int,default=60,help='每个子进程超时秒数，限15–120')
    args=parser.parse_args()
    if not 15<=args.timeout<=120:parser.error('超时范围15–120秒')
    if not (args.project_dir/'phase6_synthetic_experiment.py').is_file():parser.error('待诊断目录缺少Phase6入口')
    target=args.output_dir.resolve()
    if (target/'probe_summary.json').exists():raise FileExistsError('已有诊断结果，不覆盖')
    attempts=[]
    modes=[('normal_minimal',[],'minimal'),('isolated_minimal',['-I'],'minimal'),
           ('normal_project',[],'project'),('isolated_project',['-I'],'project'),
           ('preload_project',[],'preload_project')]
    for name,flags,mode in modes:
        for repeat in range(1,args.repeats+1):
            started=time.perf_counter()
            command=[sys.executable,*flags,'-u','-c',WORKER,mode,str(args.project_dir.resolve())]
            try:
                process=subprocess.run(command,capture_output=True,text=True,timeout=args.timeout)
                row=dict(mode=name,repeat=repeat,returncode=process.returncode,timed_out=False,
                         stdout=process.stdout,stderr=process.stderr)
            except subprocess.TimeoutExpired as error:
                row=dict(mode=name,repeat=repeat,returncode=None,timed_out=True,
                         stdout=as_text(error.stdout),stderr=as_text(error.stderr))
            row['seconds']=time.perf_counter()-started
            events=[]
            for line in row['stdout'].splitlines():
                try:
                    event=json.loads(line)
                    if isinstance(event,dict):events.append(event)
                except ValueError:
                    pass
            row['events']=events
            row['optimizer_check_passed']=row['returncode']==0 and any(e.get('stage')=='检查通过' for e in events)
            for field in ('stdout','stderr'):
                row[field+'_truncated']=len(row[field])>16000
                row[field]=row[field][-16000:]
            attempts.append(row)
            print(f'{name} 第{repeat}次：通过={row["optimizer_check_passed"]} 退出码={row["returncode"]} 耗时={row["seconds"]:.1f}秒',flush=True)
            if not row['optimizer_check_passed']:
                print(row['stdout']+row['stderr'],flush=True)
    def local_git(*values,cwd=PROJECT):
        run=subprocess.run(['git','-C',str(cwd),*values],capture_output=True,text=True,timeout=10)
        return run.stdout.strip() if run.returncode==0 else None
    result=dict(operation='Phase6 Adam启动与延迟导入隔离诊断',
        source_version=dict(commit=local_git('rev-parse','HEAD'),branch=local_git('branch','--show-current')),
        project_commit=local_git('rev-parse','HEAD',cwd=args.project_dir),
        attempts=attempts,all_optimizer_checks_ok=all(r['optimizer_check_passed'] for r in attempts),
        settings=dict(repeats=args.repeats,timeout=args.timeout,project_dir=str(args.project_dir)),
        model_loaded=False,real_or_synthetic_training=False,packages_changed=False,
        dummy_cpu_optimizer_step_only=True,
        note='主程序退出0表示收集完成，不能代替all_optimizer_checks_ok。-I不禁用所有site定制；预加载仅诊断，不修改训练入口。只输出失败regex的两个字符码及指纹，不输出完整pattern。CUDA版本非GPU实测。')
    target.mkdir(parents=True,exist_ok=True)
    with (target/'probe_summary.json').open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(f'收集完成：{target/"probe_summary.json"}；全部优化器检查通过={result["all_optimizer_checks_ok"]}',flush=True)


if __name__=='__main__':
    main()
