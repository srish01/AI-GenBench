from functools import partial

from generator_builders.elsa_d3.elsa_D3_subset import ELSA_D3_Subset_Builder

ELSA_D3_SD21_DatasetBuilder = partial(ELSA_D3_Subset_Builder, generator_id=2)


__all__ = ["ELSA_D3_SD21_DatasetBuilder"]
