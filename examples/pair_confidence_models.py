"""Training-only frozen spline comparisons and explicit saved-record audit."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,KFold,ParameterGrid
from open.pair_validation import _estimator
from pair_correlation_study import estimator
import pair_confidence_rules as rules

BASE_GRID={'curves__n_knots':[3,5],'ridge__alpha':[.1,1.,10.,100.]}
JOINT_GRID=dict(BASE_GRID,curves__joint_weight=[1.,10.,100.])


def signal_variance(c):
    return dict(noise=0.,additive=1.72,pure=c['b']**2*(1+c['rho']**2),
                mixed=1.36+1.2*c['rho']+c['b']**2*(1+c['rho']**2))[c['scenario']]


def oracle_gap(c):
    if c['scenario'] not in ('pure','mixed'):return 0.
    return c['b']**2*(1-c['rho']**2)**2/(1+c['rho']**2)/(signal_variance(c)*5/3)


def generate(c,n,seed):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,c['p']));rho=c['rho']
    x[:,1]=rho*x[:,0]+np.sqrt(1-rho*rho)*x[:,1];a,b=x[:,0],x[:,1]
    product=c['b']*(a*b-rho)
    truth=dict(noise=np.zeros(n),additive=a+.6*(b*b-1),pure=product,mixed=a+.6*b+product)[c['scenario']]
    sd=np.sqrt(signal_variance(c)*2/3) if c['scenario']!='noise' else 1.
    return pd.DataFrame(x,columns=[f'x{i+1}' for i in range(c['p'])]),truth+rng.normal(scale=sd,size=n),truth


def anchored(surface):
    a=np.asarray(surface,float)
    if a.shape!=(5,5) or not np.isfinite(a).all():raise ValueError('invalid surface')
    return a-a[:,[2]]-a[[2],:]+a[2,2]


def support(values):
    x=np.asarray(values,float);a,b=np.meshgrid(rules.GRID,rules.GRID,indexing='ij')
    counts=(np.abs(x[:,0,None,None]-a)<=.5)&(np.abs(x[:,1,None,None]-b)<=.5)
    counts=counts.sum(axis=0)
    mask=(counts>=5)&(counts[:,[2]]>=5)&(counts[[2],:]>=5)&(counts[2,2]>=5)
    return mask,counts


def surface_inputs(X,pair):
    a,b=np.meshgrid(rules.GRID,rules.GRID,indexing='ij');z=np.zeros((25,X.shape[1]))
    z[:,pair[0]]=a.ravel();z[:,pair[1]]=b.ravel()
    return pd.DataFrame(z,columns=X.columns)


def curve(model,X,train,pair,sd):
    predicted=model.predict(surface_inputs(X,pair)).reshape(5,5)
    mask,counts=support(X.iloc[train,list(pair)].to_numpy())
    return dict(predicted=predicted.tolist(),contrast=(anchored(predicted)/sd).tolist(),
                support=mask.tolist(),support_counts=counts.tolist(),training_sd=sd,grid=rules.GRID.tolist())


def tune(X,y,model,grid,inner):
    search=GridSearchCV(model,grid,cv=inner,scoring='neg_mean_squared_error',
                       n_jobs=1,error_score='raise',refit=True).fit(X,y)
    losses=[[-float(search.cv_results_[f'split{i}_test_score'][j]) for i in range(3)]
            for j in range(len(search.cv_results_['params']))]
    return search.best_estimator_,dict(search.best_params_),dict(
        candidate_parameters=list(search.cv_results_['params']),fold_losses=losses,
        selected_index=int(search.best_index_),inner_training_n=[len(a) for a,b in inner])


def compare(X,y,train,test,seed):
    train,test=list(map(int,train)),list(map(int,test));y=np.asarray(y,float)
    inner=list(KFold(3,shuffle=True,random_state=seed).split(train))
    base,params,selection=tune(X.iloc[train],y[train],_estimator(),BASE_GRID,inner)
    bp=base.predict(X.iloc[test]);observed=y[test]
    baseline=float(np.square(observed-bp).mean());reference=float(np.square(observed-y[train].mean()).mean())
    sd=float(np.std(y[train]))
    if reference<=0 or sd<=0:raise ValueError('constant risk or training outcome')
    records=[]
    for pair in rules.PAIRS:
        model,settings,trace=tune(X.iloc[train],y[train],estimator(pair),JOINT_GRID,inner)
        pred=model.predict(X.iloc[test]);mp={k:settings[k] for k in BASE_GRID}
        additive=_estimator().set_params(**mp).fit(X.iloc[train],y[train])
        ap=additive.predict(X.iloc[test]);jm=float(np.square(observed-pred).mean());am=float(np.square(observed-ap).mean())
        records.append(dict(pair=list(pair),parameters=settings,selection=trace,matched_parameters=mp,
            predicted=pred.tolist(),matched_predicted=ap.tolist(),joint_mse=jm,matched_additive_mse=am,
            overall_gain=(baseline-jm)/reference,matched_gain=(am-jm)/reference,
            marginal_gain=(baseline-am)/reference,curve=curve(model,X,train,pair,sd)))
    return dict(train_indices=train,assessment_indices=test,seed=seed,observed=observed.tolist(),
        baseline_predicted=bp.tolist(),baseline_mse=baseline,reference_mse=reference,
        baseline_parameters=params,baseline_selection=selection,pairs=records,fits=173)


def audit_selection(trace,settings,grid,inner):
    if trace['candidate_parameters']!=list(ParameterGrid(grid)) or trace['inner_training_n']!=[len(a) for a,b in inner]:
        raise ValueError('candidate grid or training sizes differ')
    a=np.asarray(trace['fold_losses'],float)
    if a.shape!=(len(list(ParameterGrid(grid))),3) or not np.isfinite(a).all() or (a<0).any():
        raise ValueError('invalid selection evidence')
    i=int(np.argmin(a.mean(axis=1)))
    if trace['selected_index']!=i or trace['candidate_parameters'][i]!=settings:
        raise ValueError('selected candidate differs from training CV evidence')


def audit_comparison(v,X,y,train,test,seed,refit=False):
    train,test=list(map(int,train)),list(map(int,test));y=np.asarray(y,float)
    if (v['train_indices']!=train or v['assessment_indices']!=test or v['seed']!=seed or
        v['fits']!=173 or len(v['pairs'])!=2 or set(train)&set(test)):
        raise ValueError('comparison identity, split or cost differs')
    observed=np.asarray(v['observed']);np.testing.assert_array_equal(observed,y[test])
    bp=np.asarray(v['baseline_predicted'],float)
    if bp.shape!=observed.shape or not np.isfinite(bp).all():raise ValueError('invalid baseline predictions')
    base=float(np.square(observed-bp).mean());ref=float(np.square(observed-y[train].mean()).mean())
    np.testing.assert_allclose([v['baseline_mse'],v['reference_mse']],[base,ref],rtol=1e-10,atol=1e-12)
    if ref<=0:raise ValueError('nonpositive reference risk')
    inner=list(KFold(3,shuffle=True,random_state=seed).split(train))
    audit_selection(v['baseline_selection'],v['baseline_parameters'],BASE_GRID,inner)
    if refit:
        model=_estimator().set_params(**v['baseline_parameters']).fit(X.iloc[train],y[train])
        np.testing.assert_allclose(model.predict(X.iloc[test]),bp,rtol=1e-9,atol=1e-9)
    for pair,r in zip(rules.PAIRS,v['pairs']):
        if r['pair']!=list(pair) or r['matched_parameters']!={k:r['parameters'][k] for k in BASE_GRID}:
            raise ValueError('pair or matched settings differ')
        audit_selection(r['selection'],r['parameters'],JOINT_GRID,inner)
        pred=np.asarray(r['predicted'],float);ap=np.asarray(r['matched_predicted'],float)
        if pred.shape!=observed.shape or ap.shape!=observed.shape or not np.isfinite([pred,ap]).all():
            raise ValueError('invalid pair predictions')
        jm=float(np.square(observed-pred).mean());am=float(np.square(observed-ap).mean())
        np.testing.assert_allclose([r['joint_mse'],r['matched_additive_mse'],r['overall_gain'],r['matched_gain'],r['marginal_gain']],
            [jm,am,(base-jm)/ref,(am-jm)/ref,(base-am)/ref],rtol=1e-10,atol=1e-12)
        c=r['curve'];sd=float(np.std(y[train]));mask,counts=support(X.iloc[train,list(pair)].to_numpy())
        if (c['grid']!=rules.GRID.tolist() or c['support']!=mask.tolist() or c['support_counts']!=counts.tolist()):
            raise ValueError('curve support or grid differs')
        np.testing.assert_allclose(c['training_sd'],sd,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(c['contrast'],anchored(c['predicted'])/sd,rtol=1e-10,atol=1e-12)
        if refit:
            model=estimator(pair).set_params(**r['parameters']).fit(X.iloc[train],y[train])
            marginal=_estimator().set_params(**r['matched_parameters']).fit(X.iloc[train],y[train])
            np.testing.assert_allclose(model.predict(X.iloc[test]),pred,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(marginal.predict(X.iloc[test]),ap,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(model.predict(surface_inputs(X,pair)).reshape(5,5),c['predicted'],rtol=1e-9,atol=1e-9)


def fold_records(comparisons,position):
    return [dict(n=len(v['observed']),reference_mse=v['reference_mse'],baseline_mse=v['baseline_mse'],
                 **{k:v['pairs'][position][k] for k in ('joint_mse','matched_additive_mse')}) for v in comparisons]
