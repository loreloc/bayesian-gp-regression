import pymc3 as pm


class GaussianProcess:
    def __init__(self, model, gp):
        self.model = model
        self.gp = gp

    def sample_conditional(self, trace, x_test, var_name='f_pred',
                           num_pred_samples=256):
        # Sample from the conditional distribution, given new points
        with self.model:
            f_pred = self.gp.conditional(var_name, x_test)
            pred_samples = pm.fast_sample_posterior_predictive(
                trace, var_names=[var_name],
                samples=num_pred_samples
            )
            pred_samples = pred_samples[var_name]
        return pred_samples
