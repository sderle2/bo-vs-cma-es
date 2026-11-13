import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from benchmark import *

class BenchmarkTests(unittest.TestCase):
    def test_posterior_matches_sklearn(self):
        rng=np.random.default_rng(1);X=rng.uniform(0,1,(20,3));y=np.sin(X[:,0]*4)
        gp=GaussianProcessRegressor(kernel=Matern(.3,nu=2.5),optimizer=None,normalize_y=True).fit(X,y)
        points=rng.uniform(0,1,(40,3))
        a,b=predict(gp,points);c,d=gp.predict(points,return_std=True)
        np.testing.assert_allclose(a,c,atol=1e-12);np.testing.assert_allclose(b,d,atol=1e-12)
    def test_known_optimum(self):
        for f in ['sphere','ellipsoid','rastrigin']:
            for d in [2,5,10]:
                fun,center=make_objective(f,d,1)
                self.assertAlmostEqual(fun(center),0.)
                self.assertGreaterEqual(fun(np.zeros(d)),0.)
    def test_ei_monte_carlo(self):
        rng=np.random.default_rng(42);mu=np.array([-.5,.5]);sd=np.array([.3,1.])
        draws=rng.normal(mu,sd,(200000,2))
        np.testing.assert_allclose(expected_improvement(mu,sd,0),np.maximum(-draws,0).mean(0),atol=.004)
    def test_budget_and_common_initial(self):
        cfg=dict(seed_base=42,n_initial=4,budget=13,refit_every=5,acquisition_starts=1,candidate_pool=16)
        results=[]
        with threadpool_limits(limits=1):
            for method in ['bo','cma','random']:
                df,_=run_one('sphere',2,0,method,cfg)
                self.assertEqual(len(df),13)
                self.assertTrue(np.all(np.diff(df.best)<=0))
                self.assertTrue(np.isfinite(df.value).all())
                results.append(df.value.to_numpy()[:4])
        np.testing.assert_allclose(results[0],results[1]);np.testing.assert_allclose(results[0],results[2])

if __name__=='__main__': unittest.main()
