import easydict
import numpy
import pytest

from ding.utils.normalizer_helper import DatasetNormalizer


# TODO(nyz): fix unittest bugs
@pytest.mark.tmp
class TestNormalizerHelper:

    def test_normalizer(self):
        x = numpy.random.randn(10)
        mean = x.mean()
        std = x.std()
        mins = x.min()
        maxs = x.max()
        normalizer = DatasetNormalizer({'plot': x}, 'GaussianNormalizer', 10)
        test = numpy.random.randn(1)
        normal_test = normalizer.normalize(test, 'plot')
        unnormal_test = normalizer.unnormalize(normal_test, 'plot')
        assert unnormal_test == test
        assert normal_test == (test - mean) / std

        normalizer = DatasetNormalizer({'plot': x}, 'LimitsNormalizer', 10)
        test = numpy.random.randn(1)
        normal_test1 = (test - mins) / (maxs - mins)
        normal_test1 = 2 * normal_test1 - 1
        normal_test = normalizer.normalize(test, 'plot')
        unnormal_test = normalizer.unnormalize(normal_test, 'plot')
        assert unnormal_test == test
        assert normal_test == normal_test1

        normalizer = DatasetNormalizer({'plot': x}, 'CDFNormalizer', 10)
        test = numpy.random.randn(1)
        normal_test = normalizer.normalize(test, 'plot')
        unnormal_test = normalizer.unnormalize(normal_test, 'plot')
        assert unnormal_test == test
