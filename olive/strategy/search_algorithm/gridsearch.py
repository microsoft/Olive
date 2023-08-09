from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner

from olive.strategy.search_algorithm.nni_sampler import NNISearchAlgorithm


# Define GridSearchAlgorithm which extends NNISearchAlgorithm
class GridSearchAlgorithm(NNISearchAlgorithm):

    name = "grid_search_algorithm"

    def _create_tuner(self):
        tuner = GridSearchTuner()
        tuner.update_search_space(self._search_space)
        return tuner
