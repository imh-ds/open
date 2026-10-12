from pathlib import Path
import sys,json,numpy as np,pandas as pd
sys.path.insert(0,"examples")
import pair_correlation_study as m
from sklearn.model_selection import GridSearchCV,KFold
from scipy.spatial import Delaunay
m.freeze()
out=Path(".superpowers/small-n-diagnosis");out.mkdir(parents=True,exist_ok=True)
c=next(c for c in m.tasks() if c["id"]=="pure-rho0.9-n80-p5-r00")
X,y,oracle,_=m.generate(c,c["n"],c["seed"]);train,test=m.splits(c)[1]
inner=list(KFold(3,shuffle=True,random_state=c["split_seed"]).split(train))
s=GridSearchCV(m.estimator((0,1)),dict(curves__n_knots=[3,5],ridge__alpha=[.1,1.,10.,100.],curves__joint_weight=list(m.WEIGHTS)),cv=inner,scoring="neg_mean_squared_error",error_score="raise",n_jobs=1).fit(X.iloc[train],y[train])
rows=[]
for i,params in enumerate(s.cv_results_["params"]):
 loss=[-float(s.cv_results_[f"split{j}_test_score"][i]) for j in range(3)]
 model=m.estimator((0,1)).set_params(**params).fit(X.iloc[train],y[train]);pred=model.predict(X.iloc[test]);T=model.named_steps["curves"].transform(model.named_steps["imputer"].transform(X.iloc[train]))
 centered=T-T.mean(axis=0);sing=np.linalg.svd(centered,compute_uv=False)
 df=float(np.square(sing).dot(1/(np.square(sing)+params["ridge__alpha"])))
 rows.append(dict(index=i,**params,mean_inner_mse=float(np.mean(loss)),inner_loss_sd=float(np.std(loss)),inner_loss_se=float(np.std(loss,ddof=1)/np.sqrt(3)),split_losses=loss,outer_mse=float(np.mean((y[test]-pred)**2)),effective_df=df))
best=int(s.best_index_);threshold=rows[best]["mean_inner_mse"]+rows[best]["inner_loss_se"]
params=s.best_params_;model=s.best_estimator_;pred=model.predict(X.iloc[test]);curves=model.named_steps["curves"];imputer=model.named_steps["imputer"];ridge=model.named_steps["ridge"]
features=curves.transform(imputer.transform(X.iloc[test]));width=curves.splines_.n_features_out_
base=curves.splines_.transform(imputer.transform(X.iloc[train]));a,b=curves._pair_blocks(base);projector=np.column_stack([np.ones(len(train)),a,b]);sing=np.linalg.svd(projector,compute_uv=False)
svd=m.estimator((0,1)).set_params(**params,ridge__solver="svd").fit(X.iloc[train],y[train]).predict(X.iloc[test])
strict=m.estimator((0,1)).set_params(**params,ridge__tol=1e-12).fit(X.iloc[train],y[train]).predict(X.iloc[test])
lo=X.iloc[train].min();hi=X.iloc[train].max();outside=((X.iloc[test]<lo)|(X.iloc[test]>hi)).to_numpy().any(axis=1)
hull=Delaunay(X.iloc[train][["x1","x2"]].to_numpy());outside_hull=hull.find_simplex(X.iloc[test][["x1","x2"]].to_numpy())<0
ranked=pd.DataFrame(dict(row=test,observed=y[test],oracle=oracle[test],predicted=pred,error_squared=(y[test]-pred)**2,marginal_component=features[:,:width]@ridge.coef_[:width],joint_component=features[:,width:]@ridge.coef_[width:],intercept=np.full(len(test),ridge.intercept_),outside_any_training_range=outside,outside_pair_hull=outside_hull,x1=X.iloc[test].x1.to_numpy(),x2=X.iloc[test].x2.to_numpy())).sort_values("error_squared",ascending=False)
summary=dict(case=c,outer_fold=1,train_n=len(train),test_n=len(test),inner_training_n=[len(a) for a,b in inner],parameters=params,inner_best_mse=rows[best]["mean_inner_mse"],one_se_threshold=threshold,within_one_se_indices=[r["index"] for r in rows if r["mean_inner_mse"]<=threshold],outer_mse=float(np.mean((y[test]-pred)**2)),svd_max_prediction_difference=float(abs(pred-svd).max()),strict_lsqr_max_prediction_difference=float(abs(pred-strict).max()),projection_singular_values=sing.tolist(),projection_rank=int(np.linalg.matrix_rank(projector)),training_ranges={k:[float(lo[k]),float(hi[k])] for k in X},worst_error_share=float(ranked.iloc[0].error_squared/ranked.error_squared.sum()))
m.io.write(out/"diagnosis.json",summary);m.io.write(out/"candidate_scores.json",rows);ranked.to_csv(out/"rows.csv",index=False)
print(summary)
print(pd.DataFrame(rows).sort_values("mean_inner_mse").head(12).to_string(index=False))
print(ranked.head(7).to_string(index=False))
