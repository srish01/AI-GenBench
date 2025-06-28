from functools import partial

from generator_builders.elsa_d3.elsa_D3_subset import ELSA_D3_Subset_Builder


ELSA_D3_SD14_DatasetBuilder = partial(ELSA_D3_Subset_Builder, generator_id=1)


__all__ = ["ELSA_D3_SD14_DatasetBuilder"]
