"""把白名单诊断快照提交到独立结果分支；不加载checkpoint或原始信号。"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_reporting import build_run_summary, report_text

BRANCH = 'server-diagnostics'
WHITE = ('fold.json', 'summary.json', 'history.jsonl', 'reference_sources.json',
         'epoch_metrics.jsonl', 'mediator_diagnostics.jsonl', 'collapse_diagnostics.jsonl',
         'epoch_confusion_matrices.npz', 'run_summary.json', 'run_report.txt', 'config.json',
         'probe_summary.json')
SECRET_KEY = re.compile(r'password|passwd|token|secret|credential|authorization|private.?key|api.?key|access.?key|ssh.?key|密码|口令|令牌|私钥', re.I)


def sanitize_text(text: str) -> str:
    """屏蔽常见凭证与本机绝对路径；不读取或收集进程环境内容。"""
    text = re.sub(r'-----BEGIN [^-]*PRIVATE KEY[^-]*-----.*?(?:-----END [^-]*PRIVATE KEY[^-]*-----|\Z)',
                  '<已屏蔽私钥>', text, flags=re.S)
    text = re.sub(r'\b(?:github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9_]+)\b', '<已屏蔽凭证>', text)
    text = re.sub(r'(?i)(https?://)[^/\s@]+@', r'\1<已屏蔽凭证>@', text)
    text = re.sub(r'(?i)\b(Bearer|Basic)\s+[A-Za-z0-9+/_.=-]+', '<已屏蔽认证>', text)
    text = re.sub(r'(?im)(\b[\w.-]*(?:password|passwd|token|secret|credential|api_key|access_key|authorization|private_key|ssh_key)[\w.-]*[\s\"\x27]*[:=]\s*)([^\r\n,;}]+)',
                  r'\1<已屏蔽敏感值>', text)
    text = re.sub(r'(?i)(--?(?:password|passwd|token|secret|api-key|private-key)(?:\s+|=))(?:\"[^\"]*\"|\x27[^\x27]*\x27|\S+)',
                  r'\1<已屏蔽敏感值>', text)
    text = re.sub(r'((?:密码|口令|令牌|私钥)\s*[:=：]\s*)[^\r\n,;}]+',r'\1<已屏蔽敏感值>',text)
    text = re.sub(r'(?<![\w:/])/(?!/)[^\s\"\x27<>|,;}]+', '<已屏蔽绝对路径>', text)
    text = re.sub(r'\b[A-Za-z]:[\\/][^\s\"\x27<>|,;}]+', '<已屏蔽绝对路径>', text)
    return text


def sanitize(value):
    if isinstance(value, dict):
        return {sanitize_text(str(k)): '<已屏蔽敏感字段>' if SECRET_KEY.search(str(k)) else sanitize(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return sanitize_text(value) if isinstance(value, str) else value


def reject_constant(value):
    raise ValueError(f'JSON包含非有限数值：{value}')


def clean_bytes(name: str, raw: bytes, limit: int) -> tuple[bytes, bool]:
    """二进制仅允许小型整数计数NPZ，并重新编码以去除ZIP附带内容。"""
    tail_omitted = False
    if name.endswith('.npz'):
        import numpy as np
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = {'epochs.npy', 'train.npy', 'test.npy', 'complete.npy'}
            if set(archive.namelist()) != names or len(archive.namelist()) != 4:
                raise ValueError('NPZ不是规定的逐轮混淆矩阵格式')
            if sum(item.file_size for item in archive.infolist()) > limit or any(item.flag_bits & 1 for item in archive.infolist()):
                raise ValueError('NPZ解压尺寸超过单文件限制')
            for item in archive.infolist():
                with archive.open(item) as stream:
                    if np.lib.format.read_magic(stream)!=(1,0):
                        raise ValueError('仅接受当前生产器的NPY 1.0计数格式')
                    shape,_,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=4096)
                    if dtype.kind not in ('i','u','b') or len(shape) not in (1,3):
                        raise ValueError('NPZ头部不是整数计数数组')
                    elements=1
                    for axis in shape:
                        if not isinstance(axis,int) or not 0<axis<=10000:
                            raise ValueError('NPZ头部维度超限')
                        elements*=axis
                    if elements*dtype.itemsize!=item.file_size-stream.tell():
                        raise ValueError('NPZ头部尺寸与实际载荷不同')
        with np.load(io.BytesIO(raw), allow_pickle=False) as values:
            arrays = {key: values[key] for key in ('epochs', 'train', 'test', 'complete')}
        n = len(arrays['epochs'])
        if not 0 < n <= 10000 or arrays['epochs'].shape != (n,) or arrays['complete'].shape != (n,):
            raise ValueError('NPZ轮次数或标志形状无效')
        for key, array in arrays.items():
            if key in ('train', 'test') and array.shape != (n, 2, 2):
                raise ValueError('NPZ包含非2×2矩阵，拒绝上传中间特征')
            if array.dtype.kind not in ('i', 'u', 'b') or (array < 0).any():
                raise ValueError('NPZ只允许非负整数计数/布尔标志')
        target = io.BytesIO(); np.savez_compressed(target, **arrays)
        return target.getvalue(), False
    text = raw.decode('utf-8')
    if name.endswith('.jsonl'):
        lines, records = text.splitlines(), []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=reject_constant)
            except ValueError:
                if i == len(lines)-1 and not text.endswith('\n'):
                    tail_omitted = True
                    continue
                raise
            records.append(json.dumps(sanitize(value), ensure_ascii=False, allow_nan=False))
        cleaned = '\n'.join(records) + ('\n' if records else '')
    elif name.endswith('.json'):
        cleaned = json.dumps(sanitize(json.loads(text, parse_constant=reject_constant)),
                             ensure_ascii=False, indent=2, allow_nan=False)
    else:
        cleaned = sanitize_text(text)
    return cleaned.encode('utf-8'), tail_omitted


def git(cwd: Path, *args: str, check=True):
    """不使用shell拼接、不交互索取Token；错误消息先脱敏。"""
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    env.setdefault('GIT_SSH_COMMAND', 'ssh -o BatchMode=yes')
    try:
        result = subprocess.run(['git', '-C', str(cwd), *args], capture_output=True,
                                text=True, timeout=60, env=env)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('Git操作超时；请检查网络/认证后重试') from error
    if check and result.returncode:
        raise RuntimeError('Git操作失败：' + sanitize_text(result.stderr.strip()))
    return result


def origin_url(repository: Path) -> str:
    url = git(repository, 'remote', 'get-url', 'origin').stdout.strip()
    if url == 'git@github.com:Cute33/NeuroFDAN.git' or url == 'git@github.com:Cute33/NeuroFDAN':
        return url
    parsed = urlsplit(url)
    if parsed.scheme == 'https' and parsed.hostname == 'github.com' and not parsed.username and not parsed.password and parsed.port in (None,443) and parsed.path.rstrip('/').removesuffix('.git').lower() == '/cute33/neurofdan' and not parsed.query and not parsed.fragment:
        return url
    raise ValueError('origin必须为Cute33/NeuroFDAN的无明文凭证GitHub HTTPS/SSH地址；请使用凭证助手或SSH agent，不把Token写进URL')


def read_snapshot(path: Path, limit: int) -> bytes:
    """拒绝符号链接/特殊文件，限制读取量，跳过读取期间发生变化的文件。"""
    if path.is_symlink():
        raise ValueError('符号链接不在上传白名单内')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('不是普通文件或超过单文件限制')
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > limit or (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError('文件过大或读取期间仍在写入，请稍后重试')
    return raw


def collect(repository: Path, run: Path, bundle: Path, args, source: dict) -> dict:
    missing, skipped, uploaded = [], [], []
    total, limit = 0, args.max_file_mib * 1024**2
    folders = [run] + sorted(p for p in run.glob('fold[0-9]*')
                             if re.fullmatch(r'fold\d+',p.name) and 1<=int(p.name[4:])<=32
                             and p.is_dir() and not p.is_symlink())
    candidates = []
    for folder in folders:
        prefix = '' if folder == run else folder.name + '/'
        for name in WHITE:
            path = folder / name
            if path.exists() or path.is_symlink():
                candidates.append((path, prefix + name))
            elif folder != run or (run / 'fold.json').exists() or name in ('run_summary.json','run_report.txt','config.json'):
                missing.append(prefix + name)
    logs = [run/'train.log']
    logs += sorted((repository/'logs').glob(f'*/{run.name}.log')) if re.fullmatch(r'[\w.-]+',run.name) else []
    logs += [Path(value).expanduser().absolute() for value in args.log]
    seen_logs = set()
    for log in logs:
        if log in seen_logs:
            continue
        seen_logs.add(log)
        if log.is_file():
            if log.suffix != '.log':
                skipped.append(dict(file='额外日志', reason='仅允许.log文本日志'))
                continue
            candidates.append((log, f'logs/train_{len(seen_logs):02d}.log'))
        elif log in [Path(value).expanduser().absolute() for value in args.log]:
            missing.append('显式指定的额外日志')
    if not any(target.startswith('logs/') for _,target in candidates):
        missing.append('train.log或对应终端日志（可用--log显式指定）')
    for path,target in candidates:
        try:
            raw = read_snapshot(path,limit)
            clean,tail = clean_bytes(path.name,raw,limit)
            if len(clean)>limit or total+len(clean)>args.max_total_mib*1024**2:
                raise ValueError('脱敏后文件或总量超过限制')
            destination = bundle/target
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(clean)
            uploaded.append(dict(file=target,bytes=len(clean),sha256=hashlib.sha256(clean).hexdigest(),tail_line_omitted=tail))
            total += len(clean)
        except (OSError,ValueError,TypeError,RuntimeError,OverflowError,zipfile.BadZipFile,ImportError) as error:
            skipped.append(dict(file=target,reason=sanitize_text(str(error))))
    if not uploaded:
        raise ValueError('没有可上传的白名单文件；请检查run目录，或通过--log指定已有日志')
    # 旧Phase4没有run_summary也能上传；重新摘要仅写到本地快照，不修改实验目录。
    try:
        report = sanitize(build_run_summary(run))
    except (OSError,ValueError,TypeError,AttributeError):
        report=dict(folds=[],formal_loso_verified=False,note='部分源文件结构异常，无法生成完整摘要；请查看manifest及原产物',
                    selection_bias='测试集参与选模的偏差仍存在')
    for name,content in [('upload_summary.json',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)),
                          ('upload_report.txt',sanitize_text(report_text(report)))]:
        data=content.encode('utf-8')
        if total+len(data) <= args.max_total_mib*1024**2 and len(data)<=limit:
            (bundle/name).write_bytes(data)
            uploaded.append(dict(file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
            total += len(data)
        else:
            skipped.append(dict(file=name,reason='摘要超过限制'))
    manifest=dict(format_version=1,created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),upload_checkout=source,
        recorded_run_sources=[dict(fold=f['fold'],source_version=f['source_version']) for f in report['folds']],
        metadata=[{key:f[key] for key in ('fold','dataset','task','seed','smoke')} for f in report['folds']],
        uploaded=uploaded,missing=missing,skipped=skipped,total_bytes=total,
        limits=dict(max_file_mib=args.max_file_mib,max_total_mib=args.max_total_mib),
        snapshot_atomic=False, note='运行中的多个文件不是事务快照；上传源码SHA不等于旧运行源码SHA。未上传模型权重/原始信号。',
        sanitization='白名单；拒绝链接；文本常见凭证/绝对路径屏蔽；数值NPZ重新编码。自动脱敏不能保证发现任意自由文本中的自定义秘密，首次建议dry-run人工检查。')
    # manifest也计入总预算；必要时从快照末尾移除文件，明确记入skipped，不上传超限包。
    while True:
        manifest['total_bytes']=total
        encoded=json.dumps(sanitize(manifest),ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
        if total+len(encoded)<=args.max_total_mib*1024**2:
            break
        if not uploaded:
            raise ValueError('仅manifest就超过总量限制')
        removed=uploaded.pop()
        (bundle/removed['file']).unlink()
        total-=removed['bytes']
        skipped.append(dict(file=removed['file'],reason='为manifest保留总量预算'))
    (bundle/'manifest.json').write_bytes(encoded)
    return report


def publish(repository: Path, url: str, bundle: Path, run_id: str) -> tuple[str, str]:
    identity = [git(repository,'config','user.'+key,check=False).stdout.strip() for key in ('name','email')]
    if not all(identity):
        raise ValueError('缺少Git作者信息；先配置git config user.name与user.email，可使用GitHub noreply邮箱')
    with tempfile.TemporaryDirectory(prefix='neurofdan-upload-git-') as temporary:
        workspace=Path(temporary)
        git(workspace,'init')
        git(workspace,'remote','add','origin',url)
        git(workspace,'config','user.name',identity[0]); git(workspace,'config','user.email',identity[1])
        for attempt in range(3):
            remote=git(workspace,'ls-remote','--heads','origin',f'refs/heads/{BRANCH}').stdout.strip()
            if remote:
                git(workspace,'fetch','--depth=1','origin',f'refs/heads/{BRANCH}')
                # 只切换临时Git工作区；失败的自有提交未发布，快照保存在外部bundle。
                git(workspace,'checkout','--detach','FETCH_HEAD')
            else:
                git(workspace,'symbolic-ref','HEAD',f'refs/heads/{BRANCH}')
            target=workspace/'runs'/run_id
            if target.is_dir() and not any(target.iterdir()):
                target.rmdir()
            if target.exists():
                raise ValueError('唯一上传目录已存在，拒绝覆盖历史')
            target.mkdir(parents=True)
            manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
            files=[]
            for entry in [*manifest['uploaded'],dict(file='manifest.json')]:
                relative=Path(entry['file'])
                if relative.is_absolute() or '..' in relative.parts or len(relative.parts)>2:
                    raise ValueError('快照清单含非法目标路径')
                allowed=(relative.name in (*WHITE,'manifest.json','upload_summary.json','upload_report.txt') or
                         (relative.parent.as_posix()=='logs' and re.fullmatch(r'train_\d+\.log',relative.name)))
                parent_allowed=(relative.parent.as_posix()=='.' or re.fullmatch(r'fold\d+',relative.parent.as_posix()) or
                                (relative.parent.as_posix()=='logs' and re.fullmatch(r'train_\d+\.log',relative.name)))
                if not allowed or not parent_allowed:
                    raise ValueError('快照清单包含非白名单文件')
                raw=read_snapshot(bundle/relative,16*1024**2)
                if entry.get('sha256') and hashlib.sha256(raw).hexdigest()!=entry['sha256']:
                    raise ValueError('快照在收集后发生变化，拒绝提交')
                destination=target/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(raw)
                files.append(destination.relative_to(workspace).as_posix())
            # 仅对清单中明确的白名单文件覆盖ignore；从不对push使用force。
            git(workspace,'add','-f','--',*files)
            git(workspace,'commit','-m',f'新增服务器诊断快照：{run_id}')
            sha=git(workspace,'rev-parse','HEAD').stdout.strip()
            pushed=git(workspace,'push','origin',f'HEAD:refs/heads/{BRANCH}',check=False)
            if pushed.returncode==0:
                observed=git(workspace,'ls-remote','--heads','origin',f'refs/heads/{BRANCH}').stdout.split()
                if observed and observed[0]==sha:
                    return sha,observed[0]
                # 并发上传者可能已继续提交；再次取远端并验证本提交仍为其祖先。
                git(workspace,'fetch','origin',f'refs/heads/{BRANCH}')
                ancestor=git(workspace,'merge-base','--is-ancestor',sha,'FETCH_HEAD',check=False)
                if ancestor.returncode==0:
                    print('远端随后有并发提交；已验证本上传提交仍在远端历史中。')
                    return sha,git(workspace,'rev-parse','FETCH_HEAD').stdout.strip()
                raise RuntimeError('push返回成功，但远端历史核验失败；请保留快照核查')
            if attempt==2 or not re.search(r'non-fast-forward|fetch first',pushed.stderr,re.I):
                raise RuntimeError('上传push失败：'+sanitize_text(pushed.stderr.strip()))
            print('远端有并发新提交；从最新远端重新追加本快照，不强推。')
    raise RuntimeError('上传未完成')


def main() -> None:
    parser=argparse.ArgumentParser(description='白名单诊断快照上传到server-diagnostics，不切换实验分支',add_help=False)
    parser.add_argument('-h','--help',action='help',help='显示中文参数说明并退出')
    parser.add_argument('run_dir',help='已有实验run目录或单fold目录')
    parser.add_argument('--log',action='append',default=[],help='额外指定一个.log终端日志，可重复')
    parser.add_argument('--dry-run',action='store_true',help='仅生成脱敏快照，保留本地目录，不联网或提交')
    parser.add_argument('--bundle_dir',help='本地快照目录，必须为空；默认用系统临时目录')
    parser.add_argument('--keep_bundle',action='store_true',help='成功上传后也保留自动生成的本地快照')
    parser.add_argument('--max_file_mib',type=int,default=8,help='单文件及NPZ解压大小上限MiB，超过则跳过')
    parser.add_argument('--max_total_mib',type=int,default=32,help='累计载荷大小上限MiB，超过文件明确跳过')
    args=parser.parse_args()
    if not 1<=args.max_file_mib<=16 or not args.max_file_mib<=args.max_total_mib<=64:
        parser.error('单文件限1–16MiB，总量不得小于单文件且不超过64MiB')
    repository=Path(__file__).resolve().parents[1]
    run=Path(args.run_dir).expanduser().absolute()
    bundle=None
    try:
        root=git(repository,'rev-parse','--show-toplevel').stdout.strip()
        if Path(root).resolve()!=repository or not run.is_dir() or run.is_symlink():
            raise ValueError('项目Git根目录或run目录无效，不接受run符号链接')
        source=dict(commit=git(repository,'rev-parse','HEAD').stdout.strip(),
                    branch=git(repository,'branch','--show-current').stdout.strip(),
                    tracked_changes=bool(git(repository,'status','--porcelain','--untracked-files=no').stdout.strip()))
        if source['branch']==BRANCH:
            raise ValueError('请从实验代码分支调用，不在server-diagnostics工作区运行')
        experiment_repository=repository
        location=git(run,'rev-parse','--show-toplevel',check=False)
        experiment=None
        if location.returncode==0:
            experiment_repository=Path(location.stdout.strip())
            experiment=dict(commit=git(experiment_repository,'rev-parse','HEAD').stdout.strip(),
                            branch=git(experiment_repository,'branch','--show-current').stdout.strip(),
                            tracked_changes=bool(git(experiment_repository,'status','--porcelain','--untracked-files=no').stdout.strip()))
        source=dict(tool_checkout=source,experiment_checkout_at_upload=experiment)
        url=origin_url(repository)
        if not args.dry_run:
            git(repository,'ls-remote','--heads','origin',f'refs/heads/{BRANCH}')
        bundle=Path(args.bundle_dir).expanduser().absolute() if args.bundle_dir else Path(tempfile.mkdtemp(prefix='neurofdan-diagnostics-'))
        if bundle.exists() and any(bundle.iterdir()):
            raise ValueError('快照目录必须为空，拒绝覆盖已有文件')
        # 不允许把快照写入待收集的实验目录，避免污染正在运行的产物。
        if bundle.resolve()==run.resolve() or run.resolve() in bundle.resolve().parents:
            raise ValueError('快照目录不得位于实验run内部')
        bundle.mkdir(parents=True,exist_ok=True)
        report=collect(experiment_repository,run,bundle,args,source)
        first=report['folds'][0] if report['folds'] else {}
        slug=lambda v: re.sub(r'[^A-Za-z0-9_-]','_',str(v or 'unknown'))[:40]
        suffix=f'fold{first["fold"]:02d}' if len(report['folds'])==1 and isinstance(first.get('fold'),int) else 'run'
        run_id=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d_%H%M%S')+'_'+slug(first.get('dataset'))+'_'+slug(first.get('task'))+'_'+suffix+'_'+uuid.uuid4().hex[:8]
        print(f'已生成脱敏快照：{bundle}；目标目录：runs/{run_id}；请查看manifest中的缺失/跳过项。')
        if args.dry_run:
            print('仅预览：未联网、未创建commit、未上传GitHub。')
            return
        sha,remote_sha=publish(repository,url,bundle,run_id)
        print(f'上传已核验：分支={BRANCH}，本次提交SHA={sha}，核验时远端SHA={remote_sha}')
        print(f'https://github.com/Cute33/NeuroFDAN/tree/{sha}/runs/{run_id}')
        print('请将上面的SHA和链接发给Codex；无需发送服务器密码或Token。')
        if not args.bundle_dir and not args.keep_bundle:
            shutil.rmtree(bundle)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        print('诊断上传未确认成功：'+sanitize_text(str(error)),file=sys.stderr)
        if bundle and bundle.is_dir():
            print(f'本地快照保留于：{bundle}；原实验分支和数据未更改。',file=sys.stderr)
        raise SystemExit(1)


if __name__=='__main__':
    main()
