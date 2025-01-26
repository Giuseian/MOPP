from ADM import *

class ADMEnsemble:
    """Ensemble of K ADM models with randomly permuted output orderings."""
    def __init__(self, num_models, output_dim, input_dim, fc_layer_params=(), latent_dim=None):
        self.num_models = num_models
        self.models = []
        self.output_orderings = []

        for _ in range(num_models):
            # Generate a random permutation of output dimensions
            ordering = np.random.permutation(output_dim)
            self.output_orderings.append(ordering)

            # Initialize a new ADM model with the permuted ordering
            model = ADM(
                output_dim=output_dim,
                input_dim=input_dim,
                fc_layer_params=fc_layer_params,
                out_reranking=ordering,
                latent_dim=latent_dim
            )
            self.models.append(model)

    def predict_behavior(self, state):
        """Predicts behavior policy using all ensemble members."""
        predictions = [model.predict_behavior(state) for model in self.models]
        return predictions

    def predict_dynamics(self, state, action):
        """Predicts dynamics using all ensemble members."""
        predictions = [model.predict_dynamics(state, action) for model in self.models]
        return predictions

    def aggregate_predictions(self, predictions):
        """Aggregates predictions from the ensemble."""
        # Extract `out_mean` from each prediction
        out_means = [pred[0] for pred in predictions]  # Select the first element of each tuple
        aggregated = torch.mean(torch.stack(out_means), dim=0)  # Aggregate over ensemble members
        return aggregated