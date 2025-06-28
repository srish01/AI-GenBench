from functools import partial

from generator_builders.elsa_d3.elsa_D3_subset import ELSA_D3_Subset_Builder


ELSA_D3_DeepFloydIF_DatasetBuilder = partial(ELSA_D3_Subset_Builder, generator_id=0)


__all__ = ["ELSA_D3_DeepFloydIF_DatasetBuilder"]
