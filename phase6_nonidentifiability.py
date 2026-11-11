"""同一SCM图下的解析非识别反例；只用标准库，不替换项目生成器。"""
import math
from fractions import Fraction as F


def counterexample() -> dict:
    """Xc=U+ec、Xa=U+ea、Mc=Xc+emc；Y*=Ma+eU+sy*ey，Y=1[Y*>0]。"""
    models = {}
    for name,t,e,noise in [('A',F(1),F(1),F(1)),('B',F(2,3),F(2),F(5,9))]:
        # Ma=t*Xa+t*Mc+ema；所有外生量独立标准正态，t在两模型均非零。
        cov = 3*t+e
        observed_var = (2*t+e)**2+3*t*t+1+noise
        do_variance = e*e+t*t+1+noise
        if cov!=4 or observed_var!=14:
            raise ArithmeticError('反例协方差恒等式失败')
        probability = .5*(1+math.erf(float(2*t)/(math.sqrt(float(do_variance))*math.sqrt(2))))
        models[name] = dict(t=str(t), confounding=str(e), y_noise_variance=str(noise),
            observed_covariance=[[2,1,4],[1,2,4],[4,4,14]],
            do_mean_at_1_1=float(2*t),do_variance=float(do_variance),do_probability_at_1_1=probability)
    return dict(models=models, same_observational_distribution=True,
        unavoidable_max_absolute_error_lower_bound=abs(models['A']['do_probability_at_1_1']-models['B']['do_probability_at_1_1'])/2,
        note='Y*仅为阈值函数辅助记号；不是新SCM节点。未知中介仅凭X/Y无法区分A/B。此解析反例不替换tanh合成机制。')
